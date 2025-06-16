import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, MultiHeadAttention, LayerNormalization,
    GlobalAveragePooling1D, GlobalMaxPooling1D, Concatenate, Add
)
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.losses import Huber  # 替代 tensorflow_addons 的 PinballLoss
from joblib import dump, load

# ===================== 数据预处理强化 =====================
folder_path = r'C:\Users\adminW\Desktop\code\DeadReckoning-main\data\error'
all_data = []
all_targets = []
sequence_length = 200

# 读取数据并增强高值样本
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, engine='openpyxl')

        features = df.iloc[:, [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]].values.astype(float)
        targets = df.iloc[:, -1].values.astype(float)

        # 数据增强：对高值样本重复采样
        high_value_mask = targets > 60
        for _ in range(3 if np.any(high_value_mask) else 1):
            for i in range(0, len(features), sequence_length):
                if i + sequence_length <= len(features):
                    all_data.append(features[i:i + sequence_length])
                    all_targets.append(targets[i + sequence_length - 1])
                    # 对高值样本额外增加两次采样
                    if targets[i + sequence_length - 1] > 60:
                        all_data.append(features[i:i + sequence_length])
                        all_targets.append(targets[i + sequence_length - 1])

X = np.array(all_data)
y_raw = np.array(all_targets)

# 异常值处理（保留99%分位数）
q99 = np.quantile(y_raw, 0.99)
valid_mask = y_raw <= q99
X = X[valid_mask]
y_raw = y_raw[valid_mask]

# 对数变换
y = np.log1p(y_raw)

# 特征标准化
scaler = StandardScaler()
num_features = X.shape[2]
X_flat = X.reshape(-1, num_features)  # 展平为 (num_samples * sequence_length, num_features)
X_flat = scaler.fit_transform(X_flat)

# 选择Top10重要特征
# 注意：这里需要将 y 扩展为与 X_flat 相同的样本数量
y_expanded = np.repeat(y, sequence_length)  # 将 y 扩展为 (num_samples * sequence_length,)
selector = SelectKBest(score_func=f_regression, k=10)
X_flat = selector.fit_transform(X_flat, y_expanded)

# 恢复原始形状
num_samples = len(X)  # 原始样本数量
X = X_flat.reshape(num_samples, sequence_length, -1)  # 恢复为 (num_samples, sequence_length, num_features)
num_features = X.shape[2]

# 目标变量归一化
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# ===================== 改进模型架构 =====================
def build_enhanced_model(input_shape):
    inputs = Input(shape=input_shape)

    # 深度可分离卷积模块
    x = Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv1D(128, 3, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.4)(x)

    # 双向LSTM
    x = Bidirectional(LSTM(256, return_sequences=True,
                           dropout=0.3, recurrent_dropout=0.2))(x)
    x = BatchNormalization()(x)

    # Transformer增强模块
    attn_output = MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = LayerNormalization()(Add()([x, attn_output]))
    # 前馈网络
    ffn = Dense(512, activation='gelu')(x)
    ffn = Dense(512)(ffn)  # 确保输出形状与 x 一致
    x = LayerNormalization()(Add()([x, ffn]))

    # 多尺度特征融合
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])

    # 输出层
    outputs = Dense(1,
                    kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.Constant(5.0))(x)

    return Model(inputs, outputs)


model = build_enhanced_model((sequence_length, num_features))


# ===================== 高级训练策略 =====================
# 自定义R²指标
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# 编译配置
model.compile(
    optimizer=Nadam(learning_rate=0.0005),  # 使用 Nadam 优化器
    loss=Huber(delta=1.0),  # 使用 Huber 损失替代 PinballLoss
    metrics=[r_squared, 'mae']
)


# 动态批次生成器
def dynamic_batch_generator(X, y, batch_size=32):
    while True:
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_idx = indices[i:i + batch_size]
            yield X[batch_idx], y[batch_idx]


# 回调函数
early_stopping = EarlyStopping(monitor='val_r_squared', patience=30,
                               mode='max', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)

# 训练模型
history = model.fit(
    dynamic_batch_generator(X_train, y_train),
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train) // 32,
    epochs=500,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# ===================== 智能校准系统 =====================
# 预测与反归一化
y_pred = model.predict(X_val)
y_pred = y_scaler.inverse_transform(y_pred).flatten()
y_val_orig = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_pred = np.expm1(y_pred)
y_val_orig = np.expm1(y_val_orig)

# 分区间校准
calibration_bins = [0, 20, 40, 60, np.inf]
calibrators = {}

for i in range(len(calibration_bins) - 1):
    low, high = calibration_bins[i], calibration_bins[i + 1]
    mask = (y_val_orig >= low) & (y_val_orig < high)
    if sum(mask) > 50:
        calibrator = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300,
            learning_rate=0.05,
            max_depth=5
        )
        calibrator.fit(y_pred[mask].reshape(-1, 1), y_val_orig[mask])
        calibrators[f"{low}-{high}"] = calibrator

# 应用校准
y_pred_calibrated = y_pred.copy()
for interval, calibrator in calibrators.items():
    low, high = map(float, interval.split('-'))
    mask = (y_pred >= low) & (y_pred < high)
    if sum(mask) > 0:
        y_pred_calibrated[mask] = calibrator.predict(y_pred[mask].reshape(-1, 1))

# ===================== 保存模型 =====================
# 1. 保存 TensorFlow 模型
model.save('dir_optimized_deep_temporal_model.h5')
print("TensorFlow 模型已保存为 optimized_deep_temporal_model.h5")

# 2. 保存 Scikit-learn 校准模型
for interval, calibrator in calibrators.items():
    dump(calibrator, f'dir_calibrator_{interval}.joblib')
    print(f"校准模型 {interval} 已保存为 calibrator_{interval}.joblib")

# 保存预处理对象
dump(scaler, 'dir_scaler.joblib')
dump(selector, 'dir_selector.joblib')
dump(y_scaler, 'dir_y_scaler.joblib')

# ===================== 高级可视化 =====================
# 设置全局字体大小
plt.rcParams.update({'font.size': 14})

# 全范围对比图
plt.figure(figsize=(10, 8))
plt.scatter(y_val_orig, y_pred_calibrated, alpha=0.6, c='dodgerblue')
plt.plot([0, 160], [0, 160], 'r--', lw=2)
plt.title(f'Full Range Comparison (R²={r2_score(y_val_orig, y_pred_calibrated):.4f})', fontsize=16)
plt.xlabel('True Value', fontsize=14)
plt.ylabel('Predicted Value', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 分区间对比图
num_bins = len(calibration_bins) - 1  # 校准区间数量
cols = 2  # 两列布局
rows = (num_bins + 1) // cols  # 动态计算行数（向上取整）

# 创建分区间对比图
plt.figure(figsize=(18, 6 * rows))  # 动态调整画布高度
for i in range(num_bins):
    plt.subplot(rows, cols, i + 1)  # 每个区间占一个子图
    low, high = calibration_bins[i], calibration_bins[i + 1]
    mask = (y_val_orig >= low) & (y_val_orig < high)

    # 仅在实际存在数据点时绘制
    if sum(mask) > 0:
        plt.scatter(y_val_orig[mask], y_pred_calibrated[mask], alpha=0.6, c='green')
        plt.plot([low, high], [low, high], 'r--', lw=2)
        plt.title(f'Range {low}-{high}', fontsize=16)
        plt.xlabel('True Value', fontsize=14)
        plt.ylabel('Predicted Value', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        plt.axis('off')  # 无数据时隐藏空子图

# 调整子图间距
plt.subplots_adjust(hspace=0.4, wspace=0.3)  # 增加子图之间的垂直和水平间距
plt.tight_layout()
plt.show()

# ===================== 绘制损失曲线 =====================
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='-')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='--')
plt.title('Training and Validation Loss', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.yscale('log')  # 使用对数刻度
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===================== 绘制误差分布直方图 =====================
errors = y_pred_calibrated - y_val_orig  # 计算误差
mean_error = np.mean(errors)  # 误差均值
std_error = np.std(errors)  # 误差标准差

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
plt.axvline(mean_error, color='red', linestyle='--', label=f'Mean Error: {mean_error:.2f}')
plt.axvline(mean_error + std_error, color='green', linestyle=':', label=f'±1 Std: {std_error:.2f}')
plt.axvline(mean_error - std_error, color='green', linestyle=':')
plt.title('Error Distribution', fontsize=16)
plt.xlabel('Prediction Error', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()