import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import sin, cos, pi
import src.lowpass as lp
import statistics
import src.computeSteps as cACC

DATA_PATH = '../data/PDR_data/'
GRAPH_PATH = '../graphs/'


# filter requirements
order = 4
fs = 50
cutoff = 2


# 数据提取函数
def pull_data(dir_name, file_name):
    f = open(dir_name + '/' + file_name + '.csv')
    Hx = []
    Hy = []
    Hz = []
    He = []
    timestamps = []
    line_counter = 0
    for line in f:
        if line_counter > 0:
            value = line.split(',')
            if len(value) > 3:
                timestamps.append(float(value[0]))
                hx = float(value[1])
                hy = float(value[2])
                hz = float(value[3])
                r = math.sqrt(hx ** 2 + hy ** 2 + hz ** 2)
                Hx.append(hx)
                Hy.append(hy)
                Hz.append(hz)
                He.append(r)
        line_counter += 1
    return np.array(Hx), np.array(Hy), np.array(Hz), np.array(He), np.array(timestamps)


# 数据去均值函数
def data_corrected(data):
    return np.array(data) - np.mean(data)


# 读取数据
mag_data = pd.read_csv(DATA_PATH + 'Magnetometer.csv')
acc_data = pd.read_csv(DATA_PATH + 'Accelerometer.csv')
gyro_data = pd.read_csv(DATA_PATH + 'Gyroscope.csv')

# 校正后的传感器数据
phone_mag = np.array([data_corrected(mag_data["X (µT)"]),
                      data_corrected(mag_data["Y (µT)"]),
                      data_corrected(mag_data["Z (µT)"])])
phone_acc = np.array([data_corrected(acc_data["X (m/s^2)"]),
                      data_corrected(acc_data["Y (m/s^2)"]),
                      data_corrected(acc_data["Z (m/s^2)"])])
pitch = gyro_data["X (rad/s)"]
roll = gyro_data["Y (rad/s)"]
yaw = gyro_data["Z (rad/s)"]

# 滤波后的传感器数据
phone_mag_filtered = np.array([lp.butter_lowpass_filter(mag_data["X (µT)"], cutoff, fs, order),
                               lp.butter_lowpass_filter(mag_data["Y (µT)"], cutoff, fs, order),
                               lp.butter_lowpass_filter(mag_data["Z (µT)"], cutoff, fs, order)])

phone_acc_filtered = np.array([lp.butter_lowpass_filter(acc_data["X (m/s^2)"], cutoff, fs, order),
                               lp.butter_lowpass_filter(acc_data["Y (m/s^2)"], cutoff, fs, order),
                               lp.butter_lowpass_filter(acc_data["Z (m/s^2)"], cutoff, fs, order)])

pitch_filtered = lp.butter_lowpass_filter(gyro_data["X (rad/s)"], cutoff, fs, order)
roll_filtered = lp.butter_lowpass_filter(gyro_data["Y (rad/s)"], cutoff, fs, order)
yaw_filtered = lp.butter_lowpass_filter(gyro_data["Z (rad/s)"], cutoff, fs, order)


# 旋转矩阵函数
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


# 转换后的地磁数据
earth_mag = np.empty(phone_mag.shape)

for i in range(mag_data.shape[0]):
    earth_mag[:, i] = R_z(yaw[i]) @ R_y(roll[i]) @ R_x(pitch[i]) @ phone_mag[:, i]

x_data, y_data, z_data = earth_mag[0], earth_mag[1], earth_mag[2]
timestamp = mag_data["Time (s)"]


# 计算方向角的函数
def compute_direction(x, y):
    if y > 0:
        return 90 - math.atan(x / y) * 180 / math.pi
    elif y < 0:
        return 270 - math.atan(x / y) * 180 / math.pi
    elif x < 0:
        return 180
    else:
        return 0


# 磁力计航向计算
def compute_compass2(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        direction = (450 - direction) * math.pi / 180  # 转换到标准正北方向
        direction = 2 * math.pi - direction  # 计算正北方向的角度
        compass.append(direction)
    return np.array(compass)


# 陀螺仪航向计算
def compute_gyro_yaw(gyro_yaw, timestamps):
    dt = np.diff(timestamps)
    dt = np.append(dt, dt[-1])  # 对齐长度
    gyro_angle = np.cumsum(gyro_yaw * dt)  # 积分计算
    return gyro_angle


# 互补滤波融合
def complementary_filter(mag_yaw, gyro_yaw, alpha=0.98):
    return alpha * (gyro_yaw + gyro_yaw[0]) + (1 - alpha) * mag_yaw


# 计算路径位置
def compute_positions_all(timestamps, directions, step_length, real_positions=None, correction_interval=200,x=0,y=0):

    positions = [(x, y)]

    for i, direction in enumerate(directions):
        if (i + 1) % correction_interval == 0:
            # 每200个时间戳后，强制修改位置为真实位置
            if real_positions is not None and i < len(real_positions):
                real_x, real_y = real_positions[i]
                x, y = real_x, real_y
            positions.append((x, y))
        else:
            # 否则按估算方式继续推算位置
            dx = step_length * sin(direction)
            dy = step_length * cos(direction)
            print(f'step_length={step_length}, direction={direction},dx={dx}，dy={dy}')
            x += dx
            y += dy
            positions.append((x, y))

    return np.array(positions)


# 绘制路径图
def plot_positions(timestamps, positions, real_positions=None):
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, y_positions, marker='o', color='b', label='Estimate path')

    if real_positions is not None:
        real_x_positions = real_positions[:, 0]
        real_y_positions = real_positions[:, 1]
        plt.plot(real_x_positions, real_y_positions, marker='x', color='r', label='Real path')

    plt.scatter(x_positions[::100], y_positions[::100], color='g', marker='o', label='Ours')
    plt.title("Dead Reckoning Path")
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.grid(True)
    plt.legend()
    plt.show()

# 计算距离误差函数
def compute_distance_error(estimated_positions, real_positions):
    distance_errors = np.sqrt((estimated_positions[:, 0] - real_positions[:, 0]) ** 2 +
                              (estimated_positions[:, 1] - real_positions[:, 1]) ** 2)
    return distance_errors

# 计算方向误差函数
def compute_direction_error(estimated_positions, real_positions):
    direction_errors = []
    for est_pos, real_pos in zip(estimated_positions, real_positions):
        est_direction = math.atan2(est_pos[1], est_pos[0]) * 180 / np.pi
        real_direction = math.atan2(real_pos[1], real_pos[0]) * 180 / np.pi
        direction_error = abs(est_direction - real_direction)
        if direction_error > 180:
            direction_error = 360 - direction_error  # 计算角度误差应小于180度
        direction_errors.append(direction_error)
    return np.array(direction_errors)

if __name__ == "__main__":
    # 步数和距离估算
    steps = np.max(cACC.compute(display_graph=0))
    dist = steps * 0.0069  # 假设步长为0.69米
    step_length = dist / steps
    print(f'step_length = {step_length}')

    # 计算磁力计航向
    mag_directions = compute_compass2(x_data, y_data)

    # 计算陀螺仪航向
    gyro_directions = compute_gyro_yaw(yaw, mag_data["Time (s)"])

    # 融合磁力计与陀螺仪航向
    fused_directions = complementary_filter(mag_directions, gyro_directions)

    # 读取真实轨迹数据
    real_trajectory = pd.read_excel(DATA_PATH + 'data.xlsx', usecols=[17, 18], engine='openpyxl')  # 第18、19列为真实轨迹
    real_positions = real_trajectory.values  # 获取真实轨迹的 x 和 y 坐标

    # 计算所有时间戳的位置信息，并进行校正
    corrected_positions = compute_positions_all(mag_data["Time (s)"], fused_directions, step_length,
                                                real_positions=real_positions,x=real_positions[0,0],y=real_positions[0,1])

    # 去除最后一条估算记录，避免长度不匹配
    corrected_positions = corrected_positions[:-1]
    # 将估算位置和真实位置保存到新的 Excel 文件
    modified_data = pd.read_excel(DATA_PATH + 'data.xlsx', engine='openpyxl')

    # 只保留数据的原始列并添加估算的位置
    modified_data['Estimated X'] = corrected_positions[:, 0]
    modified_data['Estimated Y'] = corrected_positions[:, 1]

    newpath = DATA_PATH + 'temp.xlsx'
    # 保存新的文件
    modified_data.to_excel(newpath, index=False)

    # 读取保存的估算数据文件
    modified_data = pd.read_excel(newpath, engine='openpyxl')

    # 读取估算位置和真实位置
    estimated_positions = modified_data[['Estimated X', 'Estimated Y']].values
    real_trajectory = pd.read_excel(DATA_PATH + 'data.xlsx', usecols=[17, 18], engine='openpyxl')  # 第18、19列为真实轨迹
    real_positions = real_trajectory.values  # 获取真实轨迹的 x 和 y 坐标

    # 计算距离误差和方向误差
    distance_errors = compute_distance_error(estimated_positions, real_positions)
    direction_errors = compute_direction_error(estimated_positions, real_positions)

    # 将误差添加到数据中
    modified_data['Distance Error'] = distance_errors
    modified_data['Direction Error'] = direction_errors

    # 保存到新的Excel文件
    modified_data.to_excel(DATA_PATH + '_data_with_errors.xls', index=False)

    print("已计算并保存距离误差和方向误差。")

    # 绘制路径图，添加真实轨迹
    plot_positions(mag_data["Time (s)"], corrected_positions, real_positions=real_positions)