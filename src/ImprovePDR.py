"""
修改后的完整代码示例：
1. 利用加速度信号检测步态、估计每一步的步长（Weinberg 方法）
2. 利用磁力计与陀螺仪数据计算航向（并进行互补滤波融合）
3. 对于每一步：
   - 根据当前起点、步长和航向计算出 PDR 推算的候选位置（作为该步的估计结果）；
   - 然后用真实轨迹中该步完成时刻的坐标“强制”更新起点，供下一步推算使用。
4. 将每一步的 PDR 估计结果嵌入到原始 Excel 数据中（仅在步完成时刻所在行填写，其余行保持 NaN）。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import sin, cos, pi

# 引用你已有的模块（确保路径正确）
import src.lowpass as lp
import src.computeSteps as cACC
import src.peakAccelThreshold as pat

# 数据文件路径（请根据实际情况修改）
DATA_PATH = '../data/PDR_data/'
GRAPH_PATH = '../graphs/'


###############################################################################
# 1. 步长估计及步完成时刻检测（参考 computeLength.py 的思路）
###############################################################################

def get_data(data, timestamps, start, end):
    """
    获取指定时间段内的数据
    """
    index_start = 0
    index_end = len(timestamps) - 1
    while timestamps[index_start] < start:
        index_start += 1
    while timestamps[index_end] > end:
        index_end -= 1
    return data[index_start:index_end + 1]


def weinberg_estimation(data, cst):
    """
    Weinberg 步长估计公式：
       step_length = cst * (A_max - A_min)^(1/4)
    """
    Amax = np.max(data)
    Amin = np.min(data)
    return cst * (Amax - Amin) ** (1 / 4)


def compute_step_lengths_and_times(data, timestamps, threshold):
    """
    利用峰值（或交叉点）检测分割步态，
    对每一步用 Weinberg 方法估计步长，并记录下该步完成的时刻（取区间末尾时间）

    参数：
      data: 1D 加速度信号（滤波、去均值后的幅值数据）
      timestamps: 对应的时间戳数组
      threshold: 用于峰值检测的阈值参数（此处可传 0）

    返回：
      step_lengths: 每一步的估计步长数组
      step_times: 每一步完成时刻的数组
    """
    crossings = pat.peak_accel_threshold(data, timestamps, threshold)
    num_steps = len(crossings) // 2  # 每两个交叉点视为一步
    #num_steps = len(crossings) # 每两个交叉点视为一步
    step_lengths = []
    step_times = []
    for i in range(num_steps):
        start = crossings[i][0]
        if i + 2 < len(crossings):
            end = crossings[i + 2][0]
        else:
            end = crossings[-1][0]
        segment = get_data(data, timestamps, start, end)
        # 此处常数取 0.48，可根据实际情况调整
        step_length = weinberg_estimation(segment, 0.48)
        step_lengths.append(step_length)
        print(f'step_length = {step_length}')
        # 以区间末尾时间作为该步完成时刻
        step_times.append(end)
    return np.array(step_lengths), np.array(step_times)


###############################################################################
# 2. 传感器数据预处理与航向计算
###############################################################################

def data_corrected(data):
    return np.array(data) - np.mean(data)


def R_x(x):
    return np.array([[1, 0, 0],
                     [0, cos(-x), -sin(-x)],
                     [0, sin(-x), cos(-x)]])


def R_y(y):
    return np.array([[cos(-y), 0, -sin(-y)],
                     [0, 1, 0],
                     [sin(-y), 0, cos(-y)]])


def R_z(z):
    return np.array([[cos(-z), -sin(-z), 0],
                     [sin(-z), cos(-z), 0],
                     [0, 0, 1]])


def compute_direction(x, y):
    if y > 0:
        return 90 - math.atan(x / y) * 180 / math.pi
    elif y < 0:
        return 270 - math.atan(x / y) * 180 / math.pi
    elif x < 0:
        return 180
    else:
        return 0


def compute_compass2(Hx, Hy):
    """
    根据地磁数据计算航向（单位：弧度）
    """
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        # 转换为与正北对应的角度（标准数学角度）
        direction = (450 - direction) * math.pi / 180
        direction = 2 * math.pi - direction
        compass.append(direction)
    return np.array(compass)


def compute_gyro_yaw(gyro_yaw, timestamps):
    dt = np.diff(timestamps)
    dt = np.append(dt, dt[-1])
    gyro_angle = np.cumsum(gyro_yaw * dt)
    return gyro_angle


def complementary_filter(mag_yaw, gyro_yaw, alpha=0.98):
    return alpha * (gyro_yaw + gyro_yaw[0]) + (1 - alpha) * mag_yaw


###############################################################################
# 3. 根据步态事件逐步推算位置（满足“完成一步后推算该时刻位置信息，然后利用真实坐标校正作为起点”）
###############################################################################

def compute_positions_with_steps(fused_directions, sensor_timestamps, step_lengths, step_times, real_positions,
                                 real_timestamps):
    """
    对于每一步：
      1. 根据当前起点、步长及步完成时刻对应的航向计算出 PDR 推算的候选位置；
      2. 保存该候选位置作为该步的 PDR 估计结果；
      3. 然后利用真实坐标（查找与该步完成时刻最接近的真实数据）覆盖当前起点，
         供下一步推算使用。

    参数：
      fused_directions: 融合后的航向数组，对应 sensor_timestamps（例如磁力计时间戳）
      sensor_timestamps: 用于航向的时间戳数组
      step_lengths: 每一步的估计步长数组
      step_times: 每一步完成时刻的数组
      real_positions: 真实坐标数组（例如从 Excel 中读取的真实轨迹，假设与 real_timestamps 对齐）
      real_timestamps: 真实轨迹对应的时间戳数组（此处假设与 sensor_timestamps 一致）

    返回：
      estimated_positions: 每一步 PDR 估计得到的候选位置数组
    """
    estimated_positions = []
    # 初始位置取真实轨迹第一个点
    current_position = real_positions[0]

    for i in range(len(step_lengths)):
        # 找出当前步完成时刻对应的传感器时间戳索引
        idx = np.searchsorted(sensor_timestamps, step_times[i])
        if idx >= len(fused_directions):
            idx = len(fused_directions) - 1
        heading = fused_directions[idx]
        # 根据当前起点、步长和航向计算候选位置
        candidate = (current_position[0] + step_lengths[i] * sin(heading),
                     current_position[1] + step_lengths[i] * cos(heading))
        # 保存候选位置作为本步的 PDR 估计结果
        estimated_positions.append(candidate)

        current_position = candidate  # 如果无对应真实数据，则保留估计值

    return np.array(estimated_positions)


def plot_positions(estimated_positions, real_positions):
    plt.figure(figsize=(10, 6))
    plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], marker='o', label='PDR Estimated')
    plt.plot(real_positions[:, 0], real_positions[:, 1], marker='x', label='Real Path')
    plt.title("Pedestrian Dead Reckoning Path")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(True)
    plt.show()


###############################################################################
# 4. 主程序
###############################################################################

if __name__ == "__main__":
    # ---------------------------
    # 读取传感器数据
    # ---------------------------
    mag_data = pd.read_csv(DATA_PATH + 'Magnetometer.csv')
    acc_data = pd.read_csv(DATA_PATH + 'Accelerometer.csv')
    gyro_data = pd.read_csv(DATA_PATH + 'Gyroscope.csv')

    # 读取真实轨迹数据（假设 Excel 中第18、19列为真实 X, Y 坐标）
    real_trajectory = pd.read_excel(DATA_PATH + 'data.xlsx', usecols=[17, 18], engine='openpyxl')
    real_positions = real_trajectory.values
    # 此处假设真实轨迹时间戳与传感器数据一致，以磁力计时间戳为准
    sensor_timestamps = mag_data["Time (s)"].values
    real_timestamps = sensor_timestamps  # 如有独立真实时间戳，请做相应调整

    # ---------------------------
    # 处理磁力计数据：去均值、低通滤波、转换至地磁坐标
    # ---------------------------
    phone_mag = np.array([data_corrected(mag_data["X (µT)"]),
                          data_corrected(mag_data["Y (µT)"]),
                          data_corrected(mag_data["Z (µT)"])])
    mag_order = 4;
    mag_fs = 50;
    mag_cutoff = 2
    phone_mag_filtered = np.array([
        lp.butter_lowpass_filter(mag_data["X (µT)"], mag_cutoff, mag_fs, mag_order),
        lp.butter_lowpass_filter(mag_data["Y (µT)"], mag_cutoff, mag_fs, mag_order),
        lp.butter_lowpass_filter(mag_data["Z (µT)"], mag_cutoff, mag_fs, mag_order)
    ])

    # ---------------------------
    # 处理陀螺仪数据
    # ---------------------------
    pitch = gyro_data["X (rad/s)"]
    roll = gyro_data["Y (rad/s)"]
    yaw = gyro_data["Z (rad/s)"]
    gyro_order = 4;
    gyro_fs = 50;
    gyro_cutoff = 2
    pitch_filtered = lp.butter_lowpass_filter(gyro_data["X (rad/s)"], gyro_cutoff, gyro_fs, gyro_order)
    roll_filtered = lp.butter_lowpass_filter(gyro_data["Y (rad/s)"], gyro_cutoff, gyro_fs, gyro_order)
    yaw_filtered = lp.butter_lowpass_filter(gyro_data["Z (rad/s)"], gyro_cutoff, gyro_fs, gyro_order)

    # ---------------------------
    # 将手机坐标系的磁力计数据转换到地磁坐标系
    # ---------------------------
    num_samples = mag_data.shape[0]
    earth_mag = np.empty(phone_mag.shape)
    for i in range(num_samples):
        # 这里使用原始陀螺仪数据（也可以用滤波后的数据）进行旋转
        earth_mag[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ phone_mag[:, i]
    x_data, y_data, z_data = earth_mag[0], earth_mag[1], earth_mag[2]

    # ---------------------------
    # 计算磁力计航向与陀螺仪航向，并进行互补滤波融合
    # ---------------------------
    mag_directions = compute_compass2(x_data, y_data)
    gyro_yaw = compute_gyro_yaw(yaw, sensor_timestamps)
    fused_directions = complementary_filter(mag_directions, gyro_yaw)

    # ---------------------------
    # 处理加速度计数据用于步态检测和步长估计
    # ---------------------------
    x_acc, y_acc, z_acc, r_acc, acc_timestamps = cACC.pull_data(DATA_PATH, 'Accelerometer')
    acc_order = 4;
    acc_fs = 50;
    acc_cutoff = 2
    r_filtered = lp.butter_lowpass_filter(r_acc, acc_cutoff, acc_fs, acc_order)
    r_filtered = r_filtered - np.mean(r_filtered)

    # ---------------------------
    # 利用加速度信号检测步态、估计步长及步完成时刻（Weinberg 方法）
    # ---------------------------
    # threshold 参数根据实际情况调整，此处示例传 0
    step_lengths, step_times = compute_step_lengths_and_times(r_filtered, acc_timestamps, threshold=0)
    for i, t in enumerate(step_times):
        print(f"Step {i + 1} completed at time {t}")

    # ---------------------------
    # 根据每一步，利用 PDR 推算该步完成时刻的候选位置，然后利用真实坐标校正作为下一步起点
    # ---------------------------
    estimated_positions = compute_positions_with_steps(
        fused_directions, sensor_timestamps,
        step_lengths, step_times,
        real_positions, real_timestamps
    )

    # ---------------------------
    # 将每一步的 PDR 估计结果嵌入到原始 Excel 数据中
    # ---------------------------
    # 读取原始 Excel 数据（要求其中包含时间戳列，此处假设列名为 "Time (s)"）
    modified_data = pd.read_excel(DATA_PATH + 'data.xlsx', engine='openpyxl')
    modified_data['Estimated X'] = np.nan
    modified_data['Estimated Y'] = np.nan

    if "Time (s)" in modified_data.columns:
        original_timestamps = modified_data["Time (s)"].values
    else:
        # 如果没有时间戳，则假设行序号与 sensor_timestamps 对应
        original_timestamps = sensor_timestamps

    # 对于每一步，查找原始数据中最接近该步完成时刻的行索引，将估计位置写入
    indices = np.searchsorted(original_timestamps, step_times)
    # 注意：estimated_positions 数组中依次为各步的候选位置
    for i, idx in enumerate(indices):
        if idx < len(modified_data):
            modified_data.at[idx, 'Estimated X'] = estimated_positions[i, 0]
            modified_data.at[idx, 'Estimated Y'] = estimated_positions[i, 1]

    output_file = DATA_PATH + 'data_with_estimated_positions.xlsx'
    modified_data.to_excel(output_file, index=False)
    print(f"Estimated positions embedded into original data and saved to {output_file}")

    # ---------------------------
    # 绘制路径图对比（可选）
    # ---------------------------
    plot_positions(estimated_positions, real_positions)
