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

def compute_compass2(Hx, Hy):
    compass = []
    for i in range(len(Hx)):
        direction = compute_direction(Hx[i], Hy[i])
        direction = (450 - direction) * math.pi / 180
        direction = 2 * math.pi - direction
        compass.append(direction)
    return np.array(compass)

# 计算路径位置
def compute_positions_all(timestamps, directions, step_length):
    x, y = 0, 0
    positions = [(x, y)]
    for direction in directions:
        dx = step_length * sin(direction)
        dy = step_length * cos(direction)
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

    #plt.scatter(x_positions[::100], y_positions[::100], color='g', marker='o', label='Ours')
    plt.title("Dead Reckoning Path")
    plt.xlabel("X(m)")
    plt.ylabel("Y(m)")
    plt.grid(True)
    plt.legend()
    plt.show()


def save_positions(positions, filename):
    df = pd.DataFrame(positions, columns=['x', 'y'])
    df.to_csv(filename, index=False)
    print(f"轨迹数据已保存至 {filename}")

if __name__ == "__main__":
    # 步数和距离估算
    steps = np.max(cACC.compute(display_graph=0))
    dist = steps * 0.0069  # 假设步长为0.69米
    step_length = dist / steps
    print(f'step_length = {step_length}')

    # 使用所有时间戳的方向
    directions = compute_compass2(x_data, y_data)
    for i in range(0, len(directions)):
        print(directions[i])

    # 计算所有时间戳的位置信息
    positions = compute_positions_all(mag_data["Time (s)"], directions, step_length)

    # 输出每个时间戳的位置信息
    for timestamp, pos in zip(mag_data["Time (s)"], positions):
        print(f"时间戳: {timestamp:.2f} 秒 -> 位置: x = {pos[0]:.6f}, y = {pos[1]:.6f}")

    # 读取真实轨迹数据
    real_trajectory = pd.read_excel(DATA_PATH + 'data.xlsx', usecols=[17, 18], engine='openpyxl')  # 第18、19列为真实轨迹
    real_positions = real_trajectory.values  # 获取真实轨迹的 x 和 y 坐标

    save_file = GRAPH_PATH + 'pdr_trajectory.csv'
    save_positions(positions, save_file)

    # 绘制路径图，添加真实轨迹
    plot_positions(mag_data["Time (s)"], positions, real_positions=real_positions)
