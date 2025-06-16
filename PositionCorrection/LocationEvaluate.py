import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# ----------------------------
# 辅助函数：计算 DTW 距离
def dtw_distance(s1, s2):
    """
    计算两个序列 s1 和 s2 的动态时间规整（DTW）距离，
    s1, s2 均为二维坐标的列表或数组。
    """
    n, m = len(s1), len(s2)
    dtw = np.full((n+1, m+1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = euclidean(s1[i-1], s2[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j],    # 插入
                                   dtw[i, j-1],    # 删除
                                   dtw[i-1, j-1])  # 匹配
    return dtw[n, m]

# ----------------------------
# 辅助函数：计算离散 Frechét 距离
def frechet_distance(P, Q):
    """
    计算离散 Frechét 距离，P 和 Q 为二维轨迹（列表或数组）。
    实现参考递归定义，采用缓存加速。
    """
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)
    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        dist = euclidean(P[i], Q[j])
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i-1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j-1), dist)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), dist)
        else:
            ca[i, j] = np.inf
        return ca[i, j]
    return c(n-1, m-1)

# ----------------------------
# 读取保存的结果数据（positionCorrected.xlsx）
file_path = "predictRes/positionCorrected.xlsx"
df = pd.read_excel(file_path)

# 要求文件中包含下列列名：
# 真实轨迹： 'pos_x', 'pos_y'
# 估算轨迹： 'corrected X', 'corrected Y'
# 另外还包含 'Estimated X', 'Estimated Y', 'error_dist', 'error_direction' 等列

# 若数据中存在空值可以进行剔除
df = df.dropna(subset=['pos_x','pos_y','corrected X','corrected Y'])

# 将轨迹数据构成二维数组（按照出现顺序）
true_traj = df[['pos_x', 'pos_y']].to_numpy()
est_traj = df[['corrected X', 'corrected Y']].to_numpy()

# ----------------------------
# 1. 计算位置误差类指标（点对点误差）

# 对每个点计算欧式距离误差
errors = np.linalg.norm(est_traj - true_traj, axis=1)
MPE = np.mean(errors)
RMSE = np.sqrt(np.mean(errors**2))
MaxError = np.max(errors)

print("位置误差指标：")
print(f"平均定位误差 (MPE): {MPE:.4f}")
print(f"均方根误差 (RMSE): {RMSE:.4f}")
print(f"最大误差 (Max Error): {MaxError:.4f}")

# ----------------------------
# 2. 轨迹相似性指标

# DTW 距离
dtw_dist = dtw_distance(true_traj, est_traj)
# Frechét 距离
frechet_dist = frechet_distance(true_traj.tolist(), est_traj.tolist())

print("\n轨迹相似性指标：")
print(f"DTW 距离: {dtw_dist:.4f}")
print(f"离散 Frechét 距离: {frechet_dist:.4f}")

# ----------------------------
# 3. 方向性与趋势指标
# 计算各轨迹相邻点间的航向角（单位：弧度）
def compute_headings(traj):
    # traj: (N,2)
    diffs = np.diff(traj, axis=0)
    headings = np.arctan2(diffs[:,1], diffs[:,0])
    return headings

true_headings = compute_headings(true_traj)
est_headings  = compute_headings(est_traj)
# 对应的航向误差，计算角度差并规一化到 [-pi, pi]
heading_errors = true_headings - est_headings
heading_errors = (heading_errors + np.pi) % (2*np.pi) - np.pi
mean_heading_error = np.mean(np.abs(heading_errors))  # 平均绝对角误差（弧度）
mean_heading_error_deg = np.rad2deg(mean_heading_error)  # 转换为度

print("\n方向性指标：")
print(f"平均航向误差: {mean_heading_error_deg:.4f}°")

# ----------------------------
# 4. 综合误差率指标
# ATE（Absolute Trajectory Error）通常与 MPE 相同；这里直接采用 MPE
ATE = MPE

# RPE（Relative Pose Error）：计算相邻时间点位移的误差分量
def compute_rpe(traj1, traj2):
    # 计算两条轨迹相邻点间的位移向量
    disp1 = np.diff(traj1, axis=0)
    disp2 = np.diff(traj2, axis=0)
    # 计算每一段的差异距离
    rel_errors = np.linalg.norm(disp1 - disp2, axis=1)
    return np.mean(rel_errors), np.max(rel_errors)
RPE_mean, RPE_max = compute_rpe(true_traj, est_traj)

print("\n综合误差指标：")
print(f"绝对轨迹误差 (ATE): {ATE:.4f}")
print(f"相对位姿误差 (RPE) 均值: {RPE_mean:.4f}, 最大值: {RPE_max:.4f}")

# ----------------------------
# 可视化
# ① 轨迹叠加图
plt.figure(figsize=(8, 6))
plt.plot(true_traj[:, 0], true_traj[:, 1], 'bo-', label='True Trajectory (pos_x, pos_y)')
plt.plot(est_traj[:, 0], est_traj[:, 1], 'r.-', label='Estimated Trajectory (corrected X, corrected Y)')
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Trajectory Overlay Comparison")
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

# ② 误差时间序列图（随索引变化的欧式误差）
plt.figure(figsize=(8, 4))
plt.plot(errors, 'm.-')
plt.xlabel("Sample Index")
plt.ylabel("Position Error (m)")
plt.title("Position Error over Time")
plt.grid(True)
plt.tight_layout()

# ③ 累计误差分布曲线 (CDF)
sorted_errors = np.sort(errors)
cdf = np.linspace(0, 1, len(sorted_errors))
plt.figure(figsize=(8, 4))
plt.plot(sorted_errors, cdf, 'k-', lw=2)
plt.xlabel("Position Error (m)")
plt.ylabel("Cumulative Probability")
plt.title("Cumulative Distribution of Position Error (CDF)")
plt.grid(True)
plt.tight_layout()

plt.show()
