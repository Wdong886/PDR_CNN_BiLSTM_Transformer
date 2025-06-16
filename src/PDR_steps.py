import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import os
from numpy import sin, cos, pi
import src.lowpass as lp
import src.computeSteps as cACC

DATA_PATH = '../data/PDR_data/'
GRAPH_PATH = '../graphs/'


# 读取传感器数据
mag_data = pd.read_csv(DATA_PATH + 'Magnetometer.csv')
acc_data = pd.read_csv(DATA_PATH + 'Accelerometer.csv')
gyro_data = pd.read_csv(DATA_PATH + 'Gyroscope.csv')


# 计算方向角
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


# 步态检测（检测加速度峰值）
def detect_steps(acc_data, threshold=1.2):
    acc_magnitude = np.sqrt(acc_data["X (m/s^2)"] ** 2 + acc_data["Y (m/s^2)"] ** 2 + acc_data["Z (m/s^2)"] ** 2)
    steps = []
    timestamps = acc_data["Time (s)"].values

    for i in range(1, len(acc_magnitude) - 1):
        if acc_magnitude[i] > threshold and acc_magnitude[i] > acc_magnitude[i - 1] and acc_magnitude[i] > \
                acc_magnitude[i + 1]:
            steps.append((timestamps[i], i))  # 记录时间戳和索引位置

    return steps


# 计算步长
steps_detected = detect_steps(acc_data)
num_steps = len(steps_detected)
total_distance = num_steps * 0.0069  # 假设步长 0.69 米
step_length = total_distance / num_steps if num_steps > 0 else 0.69

# 计算方向角
x_data, y_data, z_data = mag_data["X (µT)"], mag_data["Y (µT)"], mag_data["Z (µT)"]
directions = compute_compass2(x_data, y_data)

# 计算步行轨迹（每一步更新一次位置）
x, y = 0, 0
positions = [(x, y)]

print("步数 | 时间戳 (s) | x 坐标 (m) | y 坐标 (m)")
print("--------------------------------------")

for step_id, (timestamp, idx) in enumerate(steps_detected):
    direction = directions[idx]  # 取该步的方向
    dx = step_length * sin(direction)
    dy = step_length * cos(direction)
    x += dx
    y += dy
    positions.append((x, y))
    print(f"{step_id + 1:4d} | {timestamp:.2f} | {x:.6f} | {y:.6f}")

# 绘制步行轨迹
positions = np.array(positions)
plt.figure(figsize=(10, 6))
plt.plot(positions[:, 0], positions[:, 1], marker='o', color='b', label='Estimated Path')
plt.scatter(positions[:, 0], positions[:, 1], color='g', marker='o', label='Step Points')
plt.title("Dead Reckoning Path (Step-Based)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True)
plt.legend()
plt.show()
