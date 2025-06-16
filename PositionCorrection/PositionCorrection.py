import pandas as pd
import numpy as np
import math

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

    # 将修正后的极坐标转换为笛卡尔坐标
    true_x = r_true * np.cos(theta_true)
    true_y = r_true * np.sin(theta_true)

    return (true_x, true_y)

# 读取CSV文件（示例数据）
df = pd.read_csv(r"predictRes/positionCorrection_preprocess.csv")

# 预处理：清理无效行（例如清除“Estimated X”、“Estimated Y”为空的行）
df = df.dropna(subset=['Estimated X', 'Estimated Y'])

# 初始化结果列
df['corrected X'] = np.nan
df['corrected Y'] = np.nan

# 逐行计算真实坐标
for idx, row in df.iterrows():
    # 如果误差数据为空则跳过
    if pd.isnull(row['error_dist']) or pd.isnull(row['error_direction']):
        continue

    try:
        true_x, true_y = calculate_true_position(
            est_x=row['Estimated X'],
            est_y=row['Estimated Y'],
            error_dist=row['error_dist'],
            error_direction=row['error_direction']
        )
        df.at[idx, 'corrected X'] = true_x
        df.at[idx, 'corrected Y'] = true_y
    except Exception as e:
        print(f"计算第{idx}行时发生错误: {e}")
        continue

# 修改输出列顺序：将CSV文件中的第一列“pos_x”和第二列“pos_y”放到最前面
output_cols = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y',
               'error_dist', 'error_direction', 'corrected X', 'corrected Y']

# 保存结果到Excel文件（如果需要CSV格式则用to_csv）
output_path = "predictRes/positionCorrected.xlsx"
df[output_cols].to_excel(output_path, index=False)

print("计算完成，结果已保存到", output_path)
