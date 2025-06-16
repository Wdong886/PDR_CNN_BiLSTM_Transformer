import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D,
    Bidirectional, Add, Concatenate, GlobalMaxPooling1D  # 新增必要的层
)
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l2

# 读取数据
folder_path = '../../data/bag'
all_data = []
all_targets = []
sequence_length = 200

for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx'):
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path, engine='openpyxl')

        # 选择特征列
        features = df.iloc[:, [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]].values.astype(float)
        targets = df.iloc[:, -2].values.astype(float)

        # 分割序列
        for i in range(0, len(features), sequence_length):
            if i + sequence_length <= len(features):
                all_data.append(features[i:i + sequence_length])
                all_targets.append(targets[i + sequence_length - 1])

# 转换为 NumPy 数组
X = np.array(all_data)
y = np.array(all_targets)
print("------------------------")
print(len(y))
print("------------------------")

# 目标变量归一化
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# 保存目标变量归一化的参数
np.save('y_scaler_mean.npy', y_scaler.mean_)
np.save('y_scaler_scale.npy', y_scaler.scale_)

# 特征标准化
scaler = StandardScaler()
num_features = X.shape[2]
X = X.reshape(-1, num_features)
X = scaler.fit_transform(X)
X = X.reshape(-1, sequence_length, num_features)

# 保存特征标准化的参数
np.save('scaler_mean.npy', scaler.mean_)
np.save('scaler_scale.npy', scaler.scale_)


# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# 构建增强版模型
def build_enhanced_model(input_shape):
    inputs = Input(shape=input_shape)

    # 深度可分离卷积模块（根据实际需求保留标准卷积）
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
    ffn = Dense(x.shape[-1])(ffn)  # 确保输出维度匹配
    x = LayerNormalization()(Add()([x, ffn]))

    # 多尺度特征融合
    gap = GlobalAveragePooling1D()(x)
    gmp = GlobalMaxPooling1D()(x)
    x = Concatenate()([gap, gmp])

    # 输出层（带特殊初始化）
    outputs = Dense(1,
                    kernel_initializer='he_normal',
                    bias_initializer=tf.keras.initializers.Constant(5.0))(x)

    return Model(inputs, outputs)


# 构建并编译模型
model = build_enhanced_model((sequence_length, num_features))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 设置回调：EarlyStopping 与 ReduceLROnPlateau
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

# 训练模型
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=500,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 模型评估
loss, mae = model.evaluate(X_val, y_val, verbose=0)
print(f'验证集均方误差 (MSE): {loss:.4f}, 平均绝对误差 (MAE): {mae:.4f}')

# 预测 & 反归一化
y_pred = model.predict(X_val)
y_pred = y_scaler.inverse_transform(y_pred)
y_val = y_scaler.inverse_transform(y_val.reshape(-1, 1))

# 计算 R² 值
r2 = r2_score(y_val, y_pred)
print(f'验证集 R2 值: {r2:.4f}')


#保存数据
results=np.hstack((y_val,y_pred))
np.savetxt('distance_regression_result.txt',results,fmt='%.4f',header='True\tPredicted',comments='')
print("真实值与预测值对已保存到 results.txt 文件中。")

# 计算回归线（利用线性拟合）
slope, intercept = np.polyfit(y_val.flatten(), y_pred.flatten(), 1)
print(f'回归线方程: y = {slope:.4f} * x + {intercept:.4f}')

# 绘制真实值与预测值对比图，并增加回归线
plt.figure(figsize=(8, 6))
plt.scatter(y_val, y_pred, alpha=0.7, color='b', label='Predicted Value vs True Value')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], color='r', linestyle='--', label='Ideal Prediction')

# 绘制回归线
x_vals = np.linspace(min(y_val), max(y_val), 100)
y_vals_reg = slope * x_vals + intercept
plt.plot(x_vals, y_vals_reg, color='g', linestyle='-', linewidth=2, label='Regression Line')
plt.title(f'Truth vs Prediction\nR² = {r2:.4f}', fontsize=16)
plt.xlabel('Truth', fontsize=14)
plt.ylabel('Prediction', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.5)
plt.show()

# 1. 训练损失曲线（Loss Curve）
plt.figure(figsize=(12, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linewidth=2)
plt.title('Training and Validation Loss Curves', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 2. MAE 收敛曲线
plt.figure(figsize=(12, 5))
plt.plot(history.history['mae'], label='Training MAE', color='blue', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', color='red', linewidth=2)
plt.title('Training and Validation MAE Curves', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 3. 误差分布直方图
errors = y_pred.flatten() - y_val.flatten()
plt.figure(figsize=(10, 5))
plt.hist(errors, bins=50, color='blue', edgecolor='black', alpha=0.7)
plt.title('Prediction Error Distribution', fontsize=16)
plt.xlabel('Prediction Error', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# 4. 残差图（Residual Plot）
plt.figure(figsize=(10, 5))
plt.scatter(y_val, errors, alpha=0.5, color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs True Values', fontsize=16)
plt.xlabel('True Values', fontsize=14)
plt.ylabel('Residuals', fontsize=14)
plt.grid(alpha=0.3)
plt.show()

# 5. 时序片段对比图
sample_idx = 0  # 选择验证集中的一个样本
plt.figure(figsize=(12, 5))
plt.plot(y_val[sample_idx], label='True', marker='o', color='blue', linewidth=2)
plt.plot(y_pred[sample_idx], label='Predicted', marker='x', color='red', linewidth=2)
plt.title(f'Time Series Prediction Comparison (Sample {sample_idx})', fontsize=16)
plt.xlabel('Time Step', fontsize=14)
plt.ylabel('errorDis', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()

# 保存模型
model.save('distance_regression_model.h5')
