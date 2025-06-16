import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# 1. 加载模型
model_path = 'distance_regression_model.h5'
model = load_model(model_path)

# 2. 加载新数据
new_data_path = r'E:\博士论文\experiment\LSR_PDR_LSTM\PersonDeadReckoning\data\1\dan_bag1_data_with_errors.xlsx'  # 新数据路径
new_data = pd.read_excel(new_data_path, engine='openpyxl')

# 3. 数据预处理
# 选择特征列（与训练时相同）
sequence_length = 200  # 与训练时相同
num_features = 14  # 特征数量（与训练时相同）
new_features = new_data.iloc[:, [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]].values.astype(float)

# 标准化特征（使用训练时的 scaler）
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')  # 加载训练时保存的均值
scaler.scale_ = np.load('scaler_scale.npy')  # 加载训练时保存的标准差
new_features = scaler.transform(new_features)

# 填充数据
total_samples = new_features.shape[0]
padding_size = (sequence_length - (total_samples % sequence_length)) % sequence_length
if padding_size > 0:
    padding = np.zeros((padding_size, num_features))
    new_features = np.vstack((new_features, padding))

# 将数据 reshape 为模型输入的形状 (batch_size, sequence_length, num_features)
new_features = new_features.reshape(-1, sequence_length, num_features)

# 4. 进行预测
predictions = model.predict(new_features)

# 5. 反归一化预测结果
y_scaler = StandardScaler()
y_scaler.mean_ = np.load('y_scaler_mean.npy')  # 加载训练时保存的目标变量均值
y_scaler.scale_ = np.load('y_scaler_scale.npy')  # 加载训练时保存的目标变量标准差
predictions = y_scaler.inverse_transform(predictions)

# 6. 保存预测结果
predictions_df = pd.DataFrame(predictions, columns=['error_dist'])
predictions_df.to_csv('../../PositionCorrection/predictRes/distance_predict.csv', index=False)
print("预测结果已保存到 predictions.csv")