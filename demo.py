
# ===================== 公共依赖库 =====================
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean


# ===================== 方向预测模块 =====================
def run_direction_predict(input_file):
    """执行方向误差预测任务"""
    print("\n========= 开始方向误差预测 =========")

    # ===================== 参数设置 =====================
    #input_file = "../../data/1/dan_bag1_data_with_errors.xlsx"  # 输入数据路径
    sequence_length = 200  # 时序长度
    selected_columns = [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]  # 特征列索引

    # ===================== 加载预处理对象与模型 =====================
    try:
        scaler = load("regressionFinal/direction_regression/dir_scaler.joblib")
        selector = load("regressionFinal/direction_regression/dir_selector.joblib")
        y_scaler = load("regressionFinal/direction_regression/dir_y_scaler.joblib")
        model = load_model("regressionFinal/direction_regression/dir_optimized_deep_temporal_model.h5", compile=False)
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # ===================== 加载校准模型 =====================
    calibrators = {}
    for file_name in os.listdir("."):
        if file_name.startswith("dir_calibrator_") and file_name.endswith(".joblib"):
            interval = file_name.replace("dir_calibrator_", "").replace(".joblib", "")
            try:
                calibrators[interval] = load(file_name)
            except Exception as e:
                print(f"校准模型 {file_name} 加载失败: {str(e)}")

    # ===================== 数据处理 =====================
    try:
        df_new = pd.read_excel(input_file, engine="openpyxl")
        features_new = df_new.iloc[:, selected_columns].values.astype(float)

        # 构造时序数据
        all_data = []
        for i in range(0, len(features_new), sequence_length):
            if i + sequence_length <= len(features_new):
                all_data.append(features_new[i:i + sequence_length])
        X_new = np.array(all_data)

        # 数据预处理
        num_features = X_new.shape[2]
        X_new_flat = X_new.reshape(-1, num_features)
        X_new_flat = scaler.transform(X_new_flat)
        X_new_flat = selector.transform(X_new_flat)
        selected_feature_count = X_new_flat.shape[1]
        X_new = X_new_flat.reshape(X_new.shape[0], sequence_length, selected_feature_count)
    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return

    # ===================== 执行预测 =====================
    try:
        y_pred = model.predict(X_new)
        y_pred = y_scaler.inverse_transform(y_pred).flatten()
        y_pred_final = np.expm1(y_pred)  # 反对数变换

        # 分区间校准
        for interval, calibrator in calibrators.items():
            parts = interval.split("-")
            low = float(parts[0])
            high = np.inf if parts[1].lower() in ["inf", "infty"] else float(parts[1])
            mask = (y_pred_final >= low) & (y_pred_final < high)
            if np.sum(mask) > 0:
                try:
                    y_pred_final[mask] = calibrator.predict(y_pred_final[mask].reshape(-1, 1))
                except Exception as e:
                    print(f"区间 {interval} 校准失败: {str(e)}")

        # 保存结果
        os.makedirs("predictRes", exist_ok=True)
        output_csv = "predictRes/direction_predicted.csv"
        pd.DataFrame(y_pred_final, columns=["error_direction"]).to_csv(output_csv, index=False)
        print(f"方向预测完成，结果保存至 {output_csv}")

    except Exception as e:
        print(f"预测过程失败: {str(e)}")


# ===================== 距离预测模块 =====================
def run_distance_predict(input_file):
    """执行距离误差预测任务"""
    print("\n========= 开始距离误差预测 =========")

    # ===================== 参数设置 =====================
    #new_data_path = "../../data/1/dan_bag1_data_with_errors.xlsx"  # 新数据路径
    sequence_length = 200  # 时序长度
    selected_cols = [2, 3, 4, 14, 15, 16, 20, 21, 22, 23, 24, 25, 26, 27]  # 特征列索引

    # ===================== 模型加载 =====================
    try:
        model = load_model("regressionFinal/distance/distance_regression_model.h5")
        print("距离预测模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return

    # ===================== 数据加载与预处理 =====================
    try:
        # 加载新数据
        new_data = pd.read_excel(input_file, engine="openpyxl")
        new_features = new_data.iloc[:, selected_cols].values.astype(float)
        print(f"数据加载成功，共 {len(new_features)} 条样本")

        # 加载标准化参数
        scaler = StandardScaler()
        scaler.mean_ = np.load("regressionFinal/distance/scaler_mean.npy")  # 加载训练时的均值
        scaler.scale_ = np.load("regressionFinal/distance/scaler_scale.npy")  # 加载训练时的标准差

        # 标准化处理
        new_features = scaler.transform(new_features)
        print("特征标准化完成")

        # 数据填充与重塑
        num_features = new_features.shape[1]
        total_samples = new_features.shape[0]
        padding_size = (sequence_length - (total_samples % sequence_length)) % sequence_length

        if padding_size > 0:
            padding = np.zeros((padding_size, num_features))
            new_features = np.vstack((new_features, padding))
            print(f"填充 {padding_size} 条空样本以对齐时序长度")

        # 重塑为三维张量 (batch, sequence, features)
        new_features = new_features.reshape(-1, sequence_length, num_features)
        print(f"数据重塑完成，输入维度: {new_features.shape}")

    except Exception as e:
        print(f"数据处理失败: {str(e)}")
        return

    # ===================== 执行预测 =====================
    try:
        predictions = model.predict(new_features)
        print("预测计算完成")

        # 反标准化处理
        y_scaler = StandardScaler()
        y_scaler.mean_ = np.load("regressionFinal/distance/y_scaler_mean.npy")  # 目标变量均值
        y_scaler.scale_ = np.load("regressionFinal/distance/y_scaler_scale.npy")  # 目标变量标准差
        predictions = y_scaler.inverse_transform(predictions)
        print("预测结果反标准化完成")

        # 保存结果
        os.makedirs("predictRes", exist_ok=True)
        output_path = "predictRes/distance_predict.csv"
        pd.DataFrame(predictions, columns=["error_dist"]).to_csv(output_path, index=False)
        print(f"预测结果已保存至 {output_path}")

    except Exception as e:
        print(f"预测过程失败: {str(e)}")


# ===================== 预处理模块 =====================
def run_preprocessing(inputfile):
    """执行数据预处理与结果合并任务"""
    print("\n========= 开始数据预处理 =========")

    # ===================== 参数配置 =====================
    #raw_data_path = "../../data/1/dan_bag1_data_with_errors.xlsx"  # 原始数据路径
    raw_data_path = inputfile
    step_size = 200  # 采样间隔
    target_columns = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y']  # 需要提取的列

    # 预测结果文件路径
    distance_csv = "predictRes/distance_predict.csv"
    direction_csv = "predictRes/direction_predicted.csv"

    # ===================== 第一步：提取原始数据 =====================
    try:
        # 读取原始Excel文件
        df_raw = pd.read_excel(raw_data_path, sheet_name='Sheet1', engine='openpyxl')
        #print(f"成功读取原始数据，共 {len(df_raw)} 行")

        # 计算采样位置（从第200行开始，索引199）
        sample_indices = range(199, len(df_raw), step_size)
        df_samples = df_raw.iloc[sample_indices][target_columns].copy()
        df_samples.reset_index(drop=True, inplace=True)
        #print(f"提取 {len(df_samples)} 行采样数据，列: {target_columns}")

    except Exception as e:
        print(f"原始数据处理失败: {str(e)}")
        return

    # ===================== 第二步：合并预测结果 =====================
    try:
        # 检查预测结果文件是否存在
        if not os.path.exists(distance_csv):
            raise FileNotFoundError(f"距离预测文件 {distance_csv} 不存在")
        if not os.path.exists(direction_csv):
            raise FileNotFoundError(f"方向预测文件 {direction_csv} 不存在")

        # 读取CSV数据
        df_distance = pd.read_csv(distance_csv)
        df_direction = pd.read_csv(direction_csv)
        print(f"读取预测结果: 距离误差样本 {len(df_distance)} 条，方向误差样本 {len(df_direction)} 条")

        # 对齐行数（扩展到最大行数，缺失值填NaN）
        max_rows = max(len(df_samples), len(df_distance), len(df_direction))
        df_samples = df_samples.reindex(range(max_rows))
        df_distance = df_distance.reindex(range(max_rows))
        df_direction = df_direction.reindex(range(max_rows))

        # 横向合并数据
        df_merged = pd.concat([
            df_samples.add_prefix('raw_'),
            df_distance.add_prefix('dist_'),
            df_direction.add_prefix('dir_')
        ], axis=1)

        # 重命名最终输出列
        output_columns = {
            'raw_pos_x': 'pos_x',
            'raw_pos_y': 'pos_y',
            'raw_Estimated X': 'Estimated X',
            'raw_Estimated Y': 'Estimated Y',
            'dist_error_dist': 'error_dist',
            'dir_error_direction': 'error_direction'
        }
        df_merged.rename(columns=output_columns, inplace=True)

        # 保留必要列
        final_cols = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y',
                      'error_dist', 'error_direction']
        df_output = df_merged[final_cols]

    except Exception as e:
        print(f"结果合并失败: {str(e)}")
        return

    # ===================== 第三步：保存预处理结果 =====================
    try:
        os.makedirs("predictRes", exist_ok=True)
        output_path = "predictRes/positionCorrection_preprocess.csv"
        df_output.to_csv(output_path, index=False)
        # print(f"预处理完成，合并后数据保存至 {output_path}")
        # print("数据结构预览:")
        # print(df_output.head(3))

    except Exception as e:
        print(f"结果保存失败: {str(e)}")


# ===================== 位置修正模块 =====================
def run_position_correction():
    """执行位置修正计算任务"""
    print("\n========= 开始位置修正计算 =========")

    # ===================== 参数配置 =====================
    input_csv = "predictRes/positionCorrection_preprocess.csv"  # 预处理数据路径
    output_excel = "predictRes/positionCorrected.xlsx"  # 输出文件路径

    # ===================== 定义坐标修正函数 =====================
    def calculate_true_position(est_x, est_y, error_dist, error_direction):
        """
        根据估算坐标和误差参数计算真实坐标
        参数：
            est_x, est_y: 估算坐标
            error_dist: 距离误差（估算距离 - 真实距离）
            error_direction: 航向误差（单位：度）
        返回：
            (true_x, true_y) 真实坐标
        """
        # 将估算坐标转换为极坐标
        r_est = np.hypot(est_x, est_y)  # 估算向量的模长
        theta_est = math.atan2(est_y, est_x)  # 估算向量的弧度角

        # 计算真实向量的极坐标参数
        r_true = r_est - error_dist  # 修正后的模长
        theta_true = theta_est - np.deg2rad(error_direction)  # 修正后的弧度角

        # 转换为笛卡尔坐标
        true_x = r_true * np.cos(theta_true)
        true_y = r_true * np.sin(theta_true)
        return (true_x, true_y)

    # ===================== 数据加载与校验 =====================
    try:
        # 检查输入文件是否存在
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"预处理文件 {input_csv} 不存在")

        # 读取数据并校验必要列
        df = pd.read_csv(input_csv)
        required_columns = ['Estimated X', 'Estimated Y', 'error_dist', 'error_direction']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入文件缺失必要列: {missing_cols}")

        # 清理无效行
        df_clean = df.dropna(subset=['Estimated X', 'Estimated Y']).copy()
        print(f"数据加载成功，原始数据 {len(df)} 行，清理后有效数据 {len(df_clean)} 行")

        # 初始化修正列
        df_clean['corrected X'] = np.nan
        df_clean['corrected Y'] = np.nan

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # ===================== 执行坐标修正计算 =====================
    error_count = 0
    for idx, row in df_clean.iterrows():
        # 跳过误差数据为空的记录
        if pd.isnull(row['error_dist']) or pd.isnull(row['error_direction']):
            continue

        try:
            # 计算修正坐标
            true_x, true_y = calculate_true_position(
                est_x=row['Estimated X'],
                est_y=row['Estimated Y'],
                error_dist=row['error_dist'],
                error_direction=row['error_direction']
            )
            df_clean.at[idx, 'corrected X'] = true_x
            df_clean.at[idx, 'corrected Y'] = true_y
        except Exception as e:
            error_count += 1
            print(f"第 {idx} 行计算失败: {str(e)}")
            continue

    # 打印计算统计信息
    total_calculated = len(df_clean) - df_clean['corrected X'].isna().sum()
    print(f"坐标修正完成，成功计算 {total_calculated}/{len(df_clean)} 条记录")
    if error_count > 0:
        print(f"警告：共 {error_count} 条记录计算失败")

    # ===================== 结果保存 =====================
    try:
        # 调整列顺序
        output_cols = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y',
                       'error_dist', 'error_direction', 'corrected X', 'corrected Y']
        final_df = df_clean.reindex(columns=output_cols)

        # 保存到Excel
        final_df.to_excel(output_excel, index=False)
        print(f"结果已保存至 {output_excel}")

    except Exception as e:
        print(f"结果保存失败: {str(e)}")


# ===================== 评估模块 =====================
def run_evaluation():
    """执行定位效果评估任务"""
    print("\n========= 开始定位效果评估 =========")

    # ===================== 参数配置 =====================
    input_excel = "predictRes/positionCorrected.xlsx"  # 修正后数据路径

    # ===================== 辅助函数定义 =====================
    def dtw_distance(s1, s2):
        """计算动态时间规整（DTW）距离"""
        n, m = len(s1), len(s2)
        dtw = np.full((n + 1, m + 1), np.inf)
        dtw[0, 0] = 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = euclidean(s1[i - 1], s2[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
        return dtw[n, m]

    def frechet_distance(P, Q):
        """计算离散Frechét距离"""
        n, m = len(P), len(Q)
        ca = np.full((n, m), -1.0)

        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            dist = euclidean(P[i], Q[j])
            if i == 0 and j == 0:
                ca[i, j] = dist
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i - 1, 0), dist)
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j - 1), dist)
            else:
                ca[i, j] = max(min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)), dist)
            return ca[i, j]

        return c(n - 1, m - 1)

    def compute_headings(traj):
        """计算轨迹航向角序列"""
        diffs = np.diff(traj, axis=0)
        headings = np.arctan2(diffs[:, 1], diffs[:, 0])
        return headings

    # ===================== 数据加载与预处理 =====================
    try:
        # 检查输入文件
        if not os.path.exists(input_excel):
            raise FileNotFoundError(f"修正后数据文件 {input_excel} 不存在")

        # 读取数据并清理空值
        df = pd.read_excel(input_excel)
        required_cols = ['pos_x', 'pos_y', 'corrected X', 'corrected Y']
        df_clean = df.dropna(subset=required_cols)
        print(f"数据加载成功，有效样本 {len(df_clean)} 条")

        # 提取轨迹数据
        true_traj = df_clean[['pos_x', 'pos_y']].values
        est_traj = df_clean[['corrected X', 'corrected Y']].values

    except Exception as e:
        print(f"数据加载失败: {str(e)}")
        return

    # ===================== 指标计算 =====================
    try:
        # 1. 位置误差指标
        errors = np.linalg.norm(est_traj - true_traj, axis=1)
        MPE = np.mean(errors)
        RMSE = np.sqrt(np.mean(errors ** 2))
        MaxError = np.max(errors)

        # 2. 轨迹相似性指标
        dtw_dist = dtw_distance(true_traj, est_traj)
        frechet_dist = frechet_distance(true_traj.tolist(), est_traj.tolist())

        # 3. 方向性指标
        true_headings = compute_headings(true_traj)
        est_headings = compute_headings(est_traj)
        heading_errors = (true_headings - est_headings + np.pi) % (2 * np.pi) - np.pi
        mean_heading_error_deg = np.rad2deg(np.mean(np.abs(heading_errors)))

        # 4. 综合误差指标
        ATE = MPE
        disp_true = np.diff(true_traj, axis=0)
        disp_est = np.diff(est_traj, axis=0)
        RPE_errors = np.linalg.norm(disp_true - disp_est, axis=1)
        RPE_mean = np.mean(RPE_errors)
        RPE_max = np.max(RPE_errors)

    except Exception as e:
        print(f"指标计算失败: {str(e)}")
        return

    # ===================== 输出评估报告 =====================
    print("\n========= 评估结果 =========")
    print(f"【定位精度】")
    print(f"- 平均定位误差 (MPE): {MPE:.4f} m")
    print(f"- 均方根误差 (RMSE): {RMSE:.4f} m")
    print(f"- 最大误差: {MaxError:.4f} m")

    print("\n【轨迹相似性】")
    print(f"- DTW距离: {dtw_dist:.4f}")
    print(f"- Frechét距离: {frechet_dist:.4f}")

    print("\n【方向一致性】")
    print(f"- 平均航向误差: {mean_heading_error_deg:.4f}°")

    print("\n【综合误差】")
    print(f"- 绝对轨迹误差 (ATE): {ATE:.4f} m")
    print(f"- 相对位姿误差均值 (RPE): {RPE_mean:.4f} m")
    print(f"- 相对位姿误差最大值: {RPE_max:.4f} m")

    def save_trajectory(traj, filename):
        df = pd.DataFrame(traj, columns=['x', 'y'])
        df.to_csv(filename, index=False)
        print(f"轨迹数据已保存至 {filename}")

    # 保存三种轨迹
    save_trajectory(true_traj, "graphs/true_trajectory.csv")
    save_trajectory(est_traj, "graphs/corrected_trajectory.csv")



    # ===================== 可视化 =====================
    try:
        # ① 轨迹叠加图
        plt.figure(figsize=(8, 6))
        # 修正：移除plot()函数中的fontsize参数
        plt.plot(true_traj[:, 0], true_traj[:, 1], 'bo-', label='True Trajectory')
        plt.plot(est_traj[:, 0], est_traj[:, 1], 'r.-', label='Estimated Trajectory')
        # 正确设置字体大小的方式
        plt.xlabel("X Coordinate", fontsize=16)
        plt.ylabel("Y Coordinate", fontsize=16)
        plt.title("Trajectory Overlay Comparison", fontsize=18, fontweight='bold')
        # 为图例设置字体大小
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('predictRes/trajectory_comparison.png')

        # ② 误差时间序列图（随索引变化的欧式误差）
        plt.figure(figsize=(8, 4))
        plt.plot(errors, 'm.-')
        plt.xlabel("Sample Index")
        plt.ylabel("Position Error (m)")
        plt.title("Position Error over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('predictRes/Position_Error.png')

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
        plt.savefig('predictRes/CDF.png')

        plt.show()
        print("\n可视化图表已显示")

    except Exception as e:
        print(f"可视化失败: {str(e)}")

    print("\n========= 评估完成 =========")


# ===================== 主执行流程 =====================
if __name__ == "__main__":
    # 创建预测结果目录
    os.makedirs("predictRes", exist_ok=True)

    input_file = "data/error/hao_body2_data_with_errors.xlsx"  # 输入数据路径
    # 执行顺序控制
    run_direction_predict(input_file)  # 步骤1：方向误差预测
    run_distance_predict(input_file)  # 步骤2：距离误差预测
    run_preprocessing(input_file)  # 步骤3：数据预处理
    run_position_correction()  # 步骤4：位置修正
    run_evaluation()  # 步骤5：效果评估

    print("\n所有流程执行完毕！最终结果见 predictRes 目录")