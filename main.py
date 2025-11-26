# forecasting_attention_project.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
import random
import os

# ========== 1. Data generation / loading ==========

def generate_synthetic_series(n_steps=20000, freq=24, noise_std=0.1, seed=42):
    """
    Generate a multivariate time series with trend + seasonality + noise.
    Returns a pandas DataFrame with columns ['feat1', 'feat2', 'feat3', 'target'].
    """
    np.random.seed(seed)
    time = np.arange(n_steps)
    # Feature 1: base seasonality (sinusoidal)
    feat1 = np.sin(2 * np.pi * time / freq) + 0.5 * np.sin(2 * np.pi * time / (freq*7))
    # Feature 2: trend + noise
    feat2 = 0.0001 * time + 0.5 * np.cos(2 * np.pi * time / (freq*30))
    # Feature 3: another seasonal component
    feat3 = np.sin(2 * np.pi * time / (freq/2)) * 0.3
    # Target: some linear + non-linear combination + noise
    target = (0.3 * feat1 + 0.5 * feat2 + 0.2 * feat3) \
             + 0.1 * np.sin(2 * np.pi * time / freq * 3) \
             + noise_std * np.random.randn(n_steps)
    df = pd.DataFrame({
        'feat1': feat1,
        'feat2': feat2,
        'feat3': feat3,
        'target': target
    }, index=pd.RangeIndex(start=0, stop=n_steps, step=1))
    return df

df = generate_synthetic_series()

# Parameters
PAST_HISTORY = 168   # e.g., past 168 steps (1 week of hourly data if freq=24)
FUTURE_TARGET = 24   # predict 24 steps ahead
BATCH_SIZE = 64
BUFFER_SIZE = 2000

# Prepare supervised learning sequences
def make_dataset(data, past_history, future_target, target_col='target'):
    X, y = [], []
    for i in range(len(data) - past_history - future_target):
        X.append(data.iloc[i : i + past_history].values)
        y.append(data.iloc[i + past_history : i + past_history + future_target][target_col].values)
    return np.array(X), np.array(y)

scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

X, y = make_dataset(df_scaled, PAST_HISTORY, FUTURE_TARGET, target_col='target')

# Split into train / val / test
n = len(X)
train_size = int(n * 0.7)
val_size = int(n * 0.1)
test_size = n - train_size - val_size

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val     = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test   = X[train_size+val_size:], y[train_size+val_size:]

print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)


# ========== 2. Baseline: stacked LSTM ==========

def build_lstm_model(input_shape, future_target):
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.LSTM(32, return_sequences=False),
        layers.Dense(future_target)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

lstm_model = build_lstm_model(input_shape=(PAST_HISTORY, df.shape[1]), future_target=FUTURE_TARGET)
lstm_model.summary()

history_lstm = lstm_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Evaluate on test set
y_pred_lstm = lstm_model.predict(X_test)
mae_lstm  = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = mean_squared_error(y_test, y_pred_lstm, squared=False)
mape_lstm = np.mean(np.abs((y_test - y_pred_lstm) / (y_test + 1e-6))) * 100
print("LSTM Test MAE:", mae_lstm, "RMSE:", rmse_lstm, "MAPE:", mape_lstm)


# ========== 3. Custom Attention-based Encoder-Decoder ==========

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape: [encoder_outputs, decoder_hidden_state]
        self.W1 = self.add_weight(name="W1",
                                  shape=(input_shape[0][2], input_shape[0][2]),
                                  initializer="glorot_uniform",
                                  trainable=True)
        self.W2 = self.add_weight(name="W2",
                                  shape=(input_shape[1][1], input_shape[0][2]),
                                  initializer="glorot_uniform",
                                  trainable=True)
        self.V = self.add_weight(name="V",
                                 shape=(input_shape[0][2], 1),
                                 initializer="glorot_uniform",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, time_steps, hidden_size)
        # decoder_hidden: (batch_size, hidden_size)
        # Expand decoder hidden to time_steps
        hidden_with_time_axis = K.expand_dims(decoder_hidden, 1)  # (batch, 1, hidden)
        score = K.tanh(K.dot(encoder_outputs, self.W1) + K.dot(hidden_with_time_axis, self.W2))
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)  # (batch, time_steps, 1)
        context_vector = attention_weights * encoder_outputs  # broadcasting
        context_vector = K.sum(context_vector, axis=1)  # (batch, hidden_size)
        return context_vector, attention_weights

def build_attention_model(input_shape, past_history, future_target, hidden_size=64):
    # Encoder
    encoder_inputs = layers.Input(shape=input_shape, name='encoder_inputs')
    encoder_lstm = layers.LSTM(hidden_size, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

    # Decoder — we will produce all future outputs in one go, using attention on encoder states
    decoder_hidden = state_h  # shape: (batch, hidden_size)

    context_vector, attention_weights = AttentionLayer(name='attention_layer')(encoder_outputs, decoder_hidden)
    outputs = layers.Dense(future_target, name='prediction')(context_vector)

    model = models.Model(inputs=encoder_inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    # Also build a separate model to output attention weights (for visualization)
    attn_model = models.Model(inputs=encoder_inputs,
                              outputs=[outputs, attention_weights])
    return model, attn_model

attn_model, attn_visual_model = build_attention_model(
    input_shape=(PAST_HISTORY, df.shape[1]),
    past_history=PAST_HISTORY,
    future_target=FUTURE_TARGET,
    hidden_size=64
)
attn_model.summary()

history_attn = attn_model.fit(train_dataset, epochs=10, validation_data=val_dataset)

y_pred_attn = attn_model.predict(X_test)
mae_attn  = mean_absolute_error(y_test, y_pred_attn)
rmse_attn = mean_squared_error(y_test, y_pred_attn, squared=False)
mape_attn = np.mean(np.abs((y_test - y_pred_attn) / (y_test + 1e-6))) * 100
print("Attention Model Test MAE:", mae_attn, "RMSE:", rmse_attn, "MAPE:", mape_attn)


# ========== 4. Attention weight visualization ==========

def plot_attention(sample_input, attn_model, idx=0):
    """
    sample_input: shape (1, past_history, n_features)
    """
    pred, attn_weights = attn_visual_model.predict(sample_input[np.newaxis, ...])
    attn_weights = attn_weights.squeeze()  # shape: (past_history,)
    plt.figure(figsize=(10,4))
    plt.plot(attn_weights)
    plt.title(f'Attention weights for sample {idx}')
    plt.xlabel('Past time step (t-PAST_HISTORY → t)')
    plt.ylabel('Attention weight (softmax)')
    plt.show()
    return pred, attn_weights

# Example: plot attention for first test sample
plot_attention(X_test[0], attn_visual_model, idx=0)

# ========== 5. Save results ==========

if not os.path.exists("results"):
    os.makedirs("results")
# Save models
lstm_model.save("results/baseline_lstm.h5")
attn_model.save("results/attention_model.h5")
print("Models saved in ./results/")
