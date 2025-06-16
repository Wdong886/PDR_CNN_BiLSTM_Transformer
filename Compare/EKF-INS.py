import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

# ================== 数据加载与预处理 ==================
df = pd.read_excel("../data/error/hao_body2_data_with_errors.xlsx")
timestamps = df.iloc[:, 1].values  # 时间戳（单位：秒）
gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values  # 陀螺仪（rad/s）
acce_data = df[['acce_x', 'acce_y', 'acce_z']].values  # 原始加速度计（m/s²）
linacce_data = df[['linacce_x', 'linacce_y', 'linacce_z']].values  # 线性加速度（已去除重力）
grav_data = df[['grav_x', 'grav_y', 'grav_z']].values  # 重力向量

# 计算时间间隔
dt = np.diff(timestamps, prepend=timestamps[0])


# ================== EKF-INS 初始化 ==================
class EKF_INS_ZUPT:
    def __init__(self):
        # 状态向量: [px, py, pz, vx, vy, vz, q0, q1, q2, q3, bgx, bgy, bgz, bax, bay, baz]
        self.state = np.zeros(16)
        self.state[6] = 1.0  # 四元数初始化为无旋转

        # 协方差矩阵
        self.P = np.eye(16) * 1e-4

        # 噪声参数（需根据IMU型号调整）
        self.gyro_noise = 1e-4
        self.acce_noise = 1e-3
        self.gyro_bias_noise = 1e-6
        self.acce_bias_noise = 1e-5

    def predict(self, gyro, acce, dt):
        # 状态预测（基于IMU数据）
        q = self.state[6:10]
        g = np.array([0, 0, -9.81])  # 重力向量

        # 姿态更新（四元数）
        omega = gyro - self.state[10:13]  # 去除零偏
        rot = Rotation.from_rotvec(omega * dt)
        q_new = (Rotation.from_quat(q) * rot).as_quat()

        # 速度与位置更新
        a_body = acce - self.state[13:16]  # 去除加速度计零偏
        a_world = Rotation.from_quat(q_new).apply(a_body) - g
        v = self.state[3:6] + a_world * dt
        p = self.state[0:3] + self.state[3:6] * dt + 0.5 * a_world * dt ** 2

        # 更新状态
        self.state[0:3] = p
        self.state[3:6] = v
        self.state[6:10] = q_new

        # 协方差预测（简化的雅可比矩阵）
        F = np.eye(16)
        # ... 此处需添加状态转移雅可比矩阵的具体实现 ...

        Q = block_diag(
            np.eye(3) * (self.gyro_noise ** 2 * dt ** 2),
            np.eye(3) * (self.acce_noise ** 2 * dt ** 2),
            np.eye(4) * 1e-6,
            np.eye(3) * (self.gyro_bias_noise ** 2 * dt),
            np.eye(3) * (self.acce_bias_noise ** 2 * dt)
        )
        self.P = F @ self.P @ F.T + Q

    def zupt_update(self):
        # ZUPT观测模型：速度为零
        H = np.zeros((3, 16))
        H[:, 3:6] = np.eye(3)  # 正确观测速度分量 vx, vy, vz（状态索引3-5）

        R = np.eye(3) * 0.01  # 观测噪声

        # 卡尔曼增益
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新状态与协方差
        self.state += K @ (np.zeros(3) - H @ self.state)
        self.P = (np.eye(16) - K @ H) @ self.P


# ================== 主处理循环 ==================
ekf = EKF_INS_ZUPT()
trajectory = []

for i in range(len(timestamps)):
    # 预测步骤
    ekf.predict(gyro_data[i], linacce_data[i], dt[i])

    # ZUPT检测（简单方差法）
    window_size = 10
    if i > window_size:
        var = np.var(acce_data[i - window_size:i], axis=0)
        if np.all(var < 0.1):  # 调整阈值
            ekf.zupt_update()

    # 记录轨迹
    trajectory.append(ekf.state[0:3].copy())

def save_trajectory(trajectory, filename):
    df = pd.DataFrame(trajectory, columns=['x', 'y', 'z'])
    df.to_csv(filename, index=False)
    print(f"轨迹数据已保存至 {filename}")

# ================== 结果可视化与保存 ==================
trajectory = np.array(trajectory)
plt.figure(figsize=(10, 6))
plt.plot(trajectory[:, 0], trajectory[:, 1], label='EKF-INS+ZUPT')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Estimated Trajectory')
plt.legend()
plt.show()

trajectory = np.array(trajectory)
save_file = "../graphs/" +  "ekf_ins_trajectory.csv"
save_trajectory(trajectory, save_file)


# 保存结果
result_df = pd.DataFrame(trajectory, columns=['pos_x', 'pos_y', 'pos_z'])
result_df.to_excel("estimated_trajectory.xlsx", index=False)