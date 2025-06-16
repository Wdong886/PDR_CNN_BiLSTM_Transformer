import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# ===================== 参数设置 =====================
# 新数据所在的 Excel 文件路径（请根据实际情况修改）
input_file = '../../data/1/dan_bag1_data_with_errors.xlsx'

# 数据所在文件夹、序列长度及需要提取的列（需与训练时一致）
sequence_length = 200
# 训练时使用了以下索引的列，注意 Python 中索引从 0 开始
selected_columns = [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]

# ===================== 加载预处理对象与模型 =====================
# 加载训练时保存的预处理对象
scaler = load('dir_scaler.joblib')
selector = load('dir_selector.joblib')
y_scaler = load('dir_y_scaler.joblib')

# 加载保存的 TensorFlow/Keras 模型
model = load_model('dir_optimized_deep_temporal_model.h5', compile=False)
# 如果需要重新编译（例如使用自定义指标），可参考训练时的编译设置

# 加载校准模型（如果有），校准模型文件名格式： dir_calibrator_{low}-{high}.joblib
calibrators = {}
for file_name in os.listdir('.'):
    if file_name.startswith('dir_calibrator_') and file_name.endswith('.joblib'):
        # 文件名形如 dir_calibrator_0-20.joblib 或 dir_calibrator_60-inf.joblib
        interval = file_name.replace('dir_calibrator_', '').replace('.joblib', '')
        calibrators[interval] = load(file_name)

# ===================== 读取并预处理新数据 =====================
# 读取 Excel 数据（请确保文件中数据格式与训练时一致）
df_new = pd.read_excel(input_file, engine='openpyxl')

# 提取需要的特征列，注意索引应与训练时一致
features_new = df_new.iloc[:, selected_columns].values.astype(float)

# 构造时序数据：按 sequence_length 划分
all_data = []
for i in range(0, len(features_new), sequence_length):
    if i + sequence_length <= len(features_new):
        seq_data = features_new[i:i + sequence_length]
        all_data.append(seq_data)
X_new = np.array(all_data)

# 数据预处理：先展平再做标准化和特征筛选，然后恢复时序形状
num_features = X_new.shape[2]
X_new_flat = X_new.reshape(-1, num_features)
X_new_flat = scaler.transform(X_new_flat)            # 使用预训练的标准化器
X_new_flat = selector.transform(X_new_flat)            # 使用训练时筛选的重要特征
# 新特征数目可能已由选择器发生变化：
selected_feature_count = X_new_flat.shape[1]
# 恢复为 (num_samples, sequence_length, selected_feature_count)
num_samples = X_new.shape[0]
X_new = X_new_flat.reshape(num_samples, sequence_length, selected_feature_count)

# ===================== 进行预测 =====================
# 使用训练好的模型预测（预测输出为经过 y_scaler 标准化和对数变换后的结果）
y_pred = model.predict(X_new)
# 反归一化（目标变量标准化）的逆变换
y_pred = y_scaler.inverse_transform(y_pred).flatten()
# 反对数变换恢复原始数值（训练时使用了 np.log1p）
y_pred_final = np.expm1(y_pred)

# ===================== 分区间校准（如果有校准模型的话） =====================
# 遍历各个校准区间（例如 "0-20", "20-40", "40-60", "60-inf"）
for interval, calibrator in calibrators.items():
    # 根据文件名解析区间端点
    parts = interval.split('-')
    low = float(parts[0])
    # 处理上限为无限大的情况
    high = np.inf if parts[1].lower() in ['inf', 'infty'] else float(parts[1])
    # 对属于该区间的预测值进行校准
    mask = (y_pred_final >= low) & (y_pred_final < high)
    if np.sum(mask) > 0:
        # 校准模型接受输入的形状通常为二维，因此 reshape(-1, 1)
        y_pred_final[mask] = calibrator.predict(y_pred_final[mask].reshape(-1, 1))

# ===================== 保存预测结果 =====================
# 创建包含预测值的DataFrame
results_df = pd.DataFrame(y_pred_final, columns=['error_direction'])

# 保存为CSV文件（可根据需要追加原始数据）
output_csv = '../../PositionCorrection/predictRes/direction_predicted.csv'
results_df.to_csv(output_csv, index=False)
print(f"预测完成，结果已保存到 {output_csv}")

# （可选）如果需要将预测结果与原始数据对应，可合并后保存：
# combined_df = pd.concat([df_new, results_df], axis=1)
# combined_df.to_csv('combined_data_with_predictions.csv', index=False)
