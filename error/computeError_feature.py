import pandas as pd
import numpy as np

# 读取 Excel 文件
file_path = r"C:\Users\wdong\Desktop\data_with_estimated_positions.xlsx"  # 你上传的文件
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

# 预处理：确保数值列正确解析
numeric_cols = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y',
                'gyro_x', 'gyro_y', 'gyro_z',
                'acce_x', 'acce_y', 'acce_z',
                'linacce_x', 'linacce_y', 'linacce_z',
                'grav_x', 'grav_y', 'grav_z',
                'magnet_x', 'magnet_y', 'magnet_z',
                'rv_x', 'rv_y', 'rv_z']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# 初始化新列
df['距离误差'] = np.nan
df['航向误差(度)'] = np.nan

# 找到所有 Estimated X/Y 有值的行
valid_indices = df[df[['Estimated X', 'Estimated Y']].notna().all(axis=1)].index

# 结果存储
results = []

# 遍历所有有效的行对
for i in range(1, len(valid_indices)):
    prev_idx = valid_indices[i - 1]  # 前一个有效行
    current_idx = valid_indices[i]  # 当前有效行

    # 真实坐标变化
    dx_real = df.loc[current_idx, 'pos_x'] - df.loc[prev_idx, 'pos_x']
    dy_real = df.loc[current_idx, 'pos_y'] - df.loc[prev_idx, 'pos_y']
    length_real = np.sqrt(dx_real**2 + dy_real**2)

    # 估算坐标变化
    dx_est = df.loc[current_idx, 'Estimated X'] - df.loc[prev_idx, 'Estimated X']
    dy_est = df.loc[current_idx, 'Estimated Y'] - df.loc[prev_idx, 'Estimated Y']
    length_est = np.sqrt(dx_est**2 + dy_est**2)

    # 计算距离误差
    distance_error = length_est - length_real

    # 计算航向误差
    theta_real = np.arctan2(dy_real, dx_real)
    theta_est = np.arctan2(dy_est, dx_est)
    delta_theta = theta_est - theta_real

    # 归一化角度差到 [-π, π]
    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))

    # 转换为角度
    heading_error = np.degrees(delta_theta)

    # 计算 IMU 传感器的向量和
    sensor_types = ['gyro', 'acce', 'linacce', 'grav', 'magnet', 'rv']
    sensor_sums = {}

    for sensor in sensor_types:
        x_sum = df[f"{sensor}_x"].iloc[prev_idx + 1:current_idx].sum()
        y_sum = df[f"{sensor}_y"].iloc[prev_idx + 1:current_idx].sum()
        z_sum = df[f"{sensor}_z"].iloc[prev_idx + 1:current_idx].sum()

        magnitude = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)

        sensor_sums[f"{sensor}_x_sum"] = x_sum
        sensor_sums[f"{sensor}_y_sum"] = y_sum
        sensor_sums[f"{sensor}_z_sum"] = z_sum
        sensor_sums[f"{sensor}_magnitude"] = magnitude  # 计算最终的向量和模长

    # 计算第3到28列（IMU数据）的平均值、方差、标准差
    imu_columns = df.columns[2:28]  # 选取第3到28列
    imu_data = df.iloc[prev_idx + 1:current_idx][imu_columns]  # 取 prev_idx+1 到 current_idx 的数据

    imu_mean = imu_data.mean().to_dict()
    imu_var = imu_data.var().to_dict()
    imu_std = imu_data.std().to_dict()

    # 整理结果
    result = {
        **sensor_sums,  # IMU 向量和
        **{f"{col}_mean": imu_mean[col] for col in imu_columns},  # 平均值
        **{f"{col}_var": imu_var[col] for col in imu_columns},  # 方差
        **{f"{col}_std": imu_std[col] for col in imu_columns},  # 标准差
        "Estimated X": df.loc[current_idx, "Estimated X"],
        "Estimated Y": df.loc[current_idx, "Estimated Y"],
        "距离误差": distance_error,
        "航向误差(度)": heading_error
    }

    results.append(result)

# 结果保存为 DataFrame 并输出
results_df = pd.DataFrame(results)
output_file = "特征和误差计算结果.xlsx"
results_df.to_excel(output_file, index=False)

print(f"计算完成，结果已保存到 {output_file}")
