import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm,block_diag
from scipy.spatial.transform import Rotation


class UKF_INS:
    def __init__(self):
        # 状态维度 [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bgx, bgy, bgz, bax, bay, baz]
        self.n = 16
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0

        # 初始状态
        self.state = np.zeros(self.n)
        self.state[6] = 1.0  # 四元数初始化为单位四元数

        # 初始协方差
        self.P = np.eye(self.n) * 1e-4

        # 过程噪声参数
        self.gyro_noise = 1e-4
        self.acce_noise = 1e-3
        self.gyro_bias_noise = 1e-6
        self.acce_bias_noise = 1e-5

        # UKF参数
        self.lambda_ = self.alpha ** 2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)
        self.Wm = np.full(2 * self.n + 1, 1 / (2 * (self.n + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.n + self.lambda_)
        self.Wc = np.copy(self.Wm)
        self.Wc[0] += (1 - self.alpha ** 2 + self.beta)

    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state
        P_sqrt = sqrtm((self.n + self.lambda_) * self.P)

        for i in range(self.n):
            sigma_points[i + 1] = self.state + P_sqrt[:, i]
            sigma_points[self.n + i + 1] = self.state - P_sqrt[:, i]

        return sigma_points

    def process_model(self, sigma_points, gyro, acce, dt):
        new_points = np.zeros_like(sigma_points)

        for i, point in enumerate(sigma_points):
            # 解包状态
            p = point[0:3]
            v = point[3:6]
            q = point[6:10]
            bg = point[10:13]
            ba = point[13:16]

            # 姿态更新
            omega = gyro - bg
            dq = Rotation.from_rotvec(omega * dt).as_quat()
            new_q = (Rotation.from_quat(q) * Rotation.from_quat(dq)).as_quat()

            # 加速度转换到世界坐标系
            a_body = acce - ba
            a_world = Rotation.from_quat(new_q).apply(a_body) - [0, 0, -9.81]

            # 速度和位置更新
            new_v = v + a_world * dt
            new_p = p + v * dt + 0.5 * a_world * dt ** 2

            # 偏差建模为随机游走
            new_bg = bg
            new_ba = ba

            new_points[i] = np.concatenate([new_p, new_v, new_q, new_bg, new_ba])

        return new_points

    def predict(self, gyro, acce, dt):
        sigma_points = self.generate_sigma_points()
        sigma_points_pred = self.process_model(sigma_points, gyro, acce, dt)

        # 计算预测均值和协方差
        self.state = np.sum(self.Wm[:, None] * sigma_points_pred, axis=0)
        delta = sigma_points_pred - self.state
        self.P = np.sum(self.Wc[..., None, None] * np.einsum('...i,...j->...ij', delta, delta), axis=0)

        # 添加过程噪声
        Q = block_diag(
            np.eye(3) * (self.gyro_noise ** 2 * dt ** 2),
            np.eye(3) * (self.acce_noise ** 2 * dt ** 2),
            np.eye(4) * 1e-6,
            np.eye(3) * (self.gyro_bias_noise ** 2 * dt),
            np.eye(3) * (self.acce_bias_noise ** 2 * dt)
        )
        self.P += Q

    def zupt_update(self):
        # 观测模型：速度为零
        H = np.zeros((3, self.n))
        H[:, 3:6] = np.eye(3)  # 观测速度分量

        R = np.eye(3) * 0.01  # 观测噪声

        # 生成sigma点
        sigma_points = self.generate_sigma_points()

        # 观测预测
        Z = sigma_points @ H.T

        # 计算观测统计量
        z_mean = np.sum(self.Wm[:, None] * Z, axis=0)
        Pzz = np.sum(self.Wc[..., None, None] * np.einsum('...i,...j->...ij', Z - z_mean, Z - z_mean), axis=0) + R
        Pxz = np.sum(self.Wc[..., None, None] * np.einsum('...i,...j->...ij', sigma_points - self.state, Z - z_mean),
                     axis=0)

        # 卡尔曼增益
        K = Pxz @ np.linalg.inv(Pzz)

        # 状态更新
        self.state += K @ (np.zeros(3) - z_mean)
        self.P -= K @ Pzz @ K.T

        # 强制协方差对称
        self.P = 0.5 * (self.P + self.P.T)


# ================== 主处理流程 ==================
if __name__ == "__main__":
    # 数据加载
    df = pd.read_excel("../data/bag/dan_bag1_data_with_errors.xlsx")
    timestamps = df.iloc[:, 1].values
    gyro_data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
    acce_data = df[['acce_x', 'acce_y', 'acce_z']].values
    linacce_data = df[['linacce_x', 'linacce_y', 'linacce_z']].values

    # 初始化UKF
    ukf = UKF_INS()
    trajectory = []

    # 处理每个数据点
    for i in range(len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1] if i > 0 else 0

        # 预测步骤
        ukf.predict(gyro_data[i], linacce_data[i], dt)

        # ZUPT检测
        if i > 10:
            window = acce_data[i - 10:i]
            if np.linalg.norm(np.var(window, axis=0)) < 0.1:
                ukf.zupt_update()

        # 记录轨迹
        trajectory.append(ukf.state[0:3].copy())

    # 可视化
    trajectory = np.array(trajectory)
    plt.figure(figsize=(10, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='UKF-INS')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Estimated Trajectory')
    plt.legend()
    plt.show()