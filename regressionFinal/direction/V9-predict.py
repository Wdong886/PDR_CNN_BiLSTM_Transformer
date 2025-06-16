import os
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

# ===================== 修复：定义自定义指标 =====================
def r_squared(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# ===================== 加载模型和校准器 =====================
# 加载 TensorFlow 模型时注册自定义指标
model = tf.keras.models.load_model(
    'dir_optimized_deep_temporal_model.h5',
    custom_objects={'r_squared': r_squared}  # 关键修复
)

# 加载 Scikit-learn 校准模型
calibrators = {}
calibration_bins = [0, 20, 40, 60, np.inf]
for i in range(len(calibration_bins) - 1):
    low, high = calibration_bins[i], calibration_bins[i + 1]
    calibrator_path = f'dir_calibrator_{low}-{high}.joblib'
    calibrators[f"{low}-{high}"] = load(calibrator_path)

# ===================== 加载预处理对象 =====================
# 加载训练时保存的预处理对象（需确保这些文件存在）
scaler = load('dir_scaler.joblib')        # 特征标准化器
selector = load('dir_selector.joblib')    # 特征选择器
y_scaler = load('dir_y_scaler.joblib')    # 目标变量标准化器

# ===================== 加载新数据并进行预处理 =====================
new_data = pd.read_excel(r'../../data/1/dan_bag1_data_with_errors.xlsx', engine='openpyxl')

# 选择相同的特征列
features = new_data.iloc[:, [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]].values.astype(float)

# 将数据分割成序列
sequence_length = 200
new_sequences = [features[i:i+sequence_length] for i in range(0, len(features), sequence_length)]
new_sequences = np.array([seq for seq in new_sequences if len(seq) == sequence_length])

# 使用训练时的标准化器和特征选择器（不要重新fit！）
new_sequences_flat = new_sequences.reshape(-1, new_sequences.shape[2])
new_sequences_flat = scaler.transform(new_sequences_flat)  # 使用transform而不是fit_transform
new_sequences_flat = selector.transform(new_sequences_flat)  # 使用训练时的特征选择器

# 恢复形状
new_sequences = new_sequences_flat.reshape(len(new_sequences), sequence_length, -1)

# ===================== 使用模型进行预测 =====================
y_pred_log = model.predict(new_sequences)

# 使用训练时的目标变量标准化器进行反归一化
y_pred = y_scaler.inverse_transform(y_pred_log).flatten()
y_pred = np.expm1(y_pred)  # 反转对数变换

# ===================== 应用校准模型 =====================
y_pred_calibrated = y_pred.copy()
for interval, calibrator in calibrators.items():
    low, high = map(float, interval.split('-'))
    mask = (y_pred >= low) & (y_pred < high)
    if sum(mask) > 0:
        y_pred_calibrated[mask] = calibrator.predict(y_pred[mask].reshape(-1, 1))

# ===================== 保存和可视化结果 =====================
output_df = pd.DataFrame({
    'Original_Index': np.arange(len(y_pred_calibrated)),
    'Predicted_Value': y_pred_calibrated
})
output_df.to_csv('predicted_direction_values.csv', index=False)

plt.figure(figsize=(12, 6))
plt.plot(y_pred_calibrated, label='Calibrated Predictions', alpha=0.7)
plt.title(f'Final Predictions (Max={np.max(y_pred_calibrated):.2f})', fontsize=14)
plt.xlabel('Sequence Index', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('predictions_visualization.png', dpi=300)
plt.show()