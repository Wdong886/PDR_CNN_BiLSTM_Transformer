import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, Add, LayerNormalization, Dense, GlobalAveragePooling1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import Huber
from joblib import dump

# ===================== 数据预处理 =====================
folder_path = '../../data/bag'
all_data = []
all_targets = []
sequence_length = 200

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, engine='openpyxl')

        features = df.iloc[:, [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]].values.astype(float)
        targets = df.iloc[:, -1].values.astype(float)

        high_value_mask = targets > 60
        for _ in range(3 if np.any(high_value_mask) else 1):
            for i in range(0, len(features), sequence_length):
                if i + sequence_length <= len(features):
                    all_data.append(features[i:i + sequence_length])
                    all_targets.append(targets[i + sequence_length - 1])
                    if targets[i + sequence_length - 1] > 60:
                        all_data.append(features[i:i + sequence_length])
                        all_targets.append(targets[i + sequence_length - 1])

X = np.array(all_data)
y_raw = np.array(all_targets)
q99 = np.quantile(y_raw, 0.99)
valid_mask = y_raw <= q99
X = X[valid_mask]
y_raw = y_raw[valid_mask]
y = np.log1p(y_raw)

scaler = StandardScaler()
num_features = X.shape[2]
X_flat = X.reshape(-1, num_features)
X_flat = scaler.fit_transform(X_flat)
y_expanded = np.repeat(y, sequence_length)
selector = SelectKBest(score_func=f_regression, k=10)
X_flat = selector.fit_transform(X_flat, y_expanded)
num_samples = len(X)
X = X_flat.reshape(num_samples, sequence_length, -1)
num_features = X.shape[2]

y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ===================== 模型构建：仅使用 Transformer =====================
def build_model_transformer(input_shape):
    inputs = Input(shape=input_shape)  # (200, 10)
    attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(inputs, inputs)
    x = Add()([inputs, attn_output])
    x = LayerNormalization()(x)

    ffn = Dense(512, activation='gelu')(x)
    ffn = Dense(input_shape[-1])(ffn)  # 调整输出维度匹配 inputs.shape[-1] = 10
    x = Add()([x, ffn])
    x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.Constant(5.0))(x)
    return Model(inputs, outputs)


input_shape = (sequence_length, num_features)
model = build_model_transformer(input_shape)


def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


model.compile(optimizer=Nadam(learning_rate=0.0005),
              loss=Huber(delta=1.0),
              metrics=[r_squared, 'mae'])
model.summary()


def dynamic_batch_generator(X, y, batch_size=32):
    while True:
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i + batch_size]
            yield X[batch_idx], y[batch_idx]


early_stopping = EarlyStopping(monitor='val_r_squared', patience=30, mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

history = model.fit(dynamic_batch_generator(X_train, y_train),
                    validation_data=(X_val, y_val),
                    steps_per_epoch=len(X_train) // 32,
                    epochs=500,
                    callbacks=[early_stopping, reduce_lr],
                    verbose=1)

# ===================== 智能校准 =====================
y_pred = model.predict(X_val)
y_pred = y_scaler.inverse_transform(y_pred).flatten()
y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_pred = np.expm1(y_pred)
y_val_orig = np.expm1(y_val_orig)

calibration_bins = [0, 20, 40, 60, np.inf]
calibrators = {}
for i in range(len(calibration_bins) - 1):
    low, high = calibration_bins[i], calibration_bins[i + 1]
    mask = (y_val_orig >= low) & (y_val_orig < high)
    if np.sum(mask) > 50:
        calibrator = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300,
                                                   learning_rate=0.05, max_depth=5)
        calibrator.fit(y_pred[mask].reshape(-1, 1), y_val_orig[mask])
        calibrators[f"{low}-{high}"] = calibrator

y_pred_calibrated = y_pred.copy()
for interval, calibrator in calibrators.items():
    low, high = map(float, interval.split('-'))
    mask = (y_pred >= low) & (y_pred < high)
    if np.sum(mask) > 0:
        y_pred_calibrated[mask] = calibrator.predict(y_pred[mask].reshape(-1, 1))

model.save('dir_model_transformer.h5')
for interval, calibrator in calibrators.items():
    dump(calibrator, f'dir_calibrator_{interval}.joblib')
dump(scaler, 'dir_scaler.joblib')
dump(selector, 'dir_selector.joblib')
dump(y_scaler, 'dir_y_scaler.joblib')


mse = mean_squared_error(y_val_orig, y_pred_calibrated)
mae = mean_absolute_error(y_val_orig, y_pred_calibrated)
r2 = r2_score(y_val_orig, y_pred_calibrated)
print(f"验证集 MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
