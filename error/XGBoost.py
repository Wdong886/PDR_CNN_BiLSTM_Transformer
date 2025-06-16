import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import os

# 定义文件夹路径
data_folder = r"D:\博士\IMU定位\final\PersonDeadReckoning\data\1"  # 存放多个 Excel 文件的文件夹路径

# 1. 读取文件夹中的所有 Excel 文件并合并数据
all_data = []

# 遍历文件夹中的所有文件
for file_name in os.listdir(data_folder):
    if not file_name.endswith(".xlsx"):
        continue  # 跳过非 Excel 文件

    # 构建文件路径
    file_path = os.path.join(data_folder, file_name)

    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

    # 将所有数据转为数值型，处理缺失值（如果有）
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()  # 简单处理：删除含缺失值的行；也可采用插值等方法

    # 将数据添加到列表中
    all_data.append(df)

# 合并所有数据
combined_df = pd.concat(all_data, axis=0, ignore_index=True)

# 2. 分离特征与目标
# 特征：前102列
X = combined_df.iloc[:, :102].values
# 目标：最后一、二列（注意：索引 -1表示方向误差；-2表示距离误差）
y = combined_df.iloc[:, -2].values

# 3. 数据预处理
# 对于树模型，标准化有时不是必须的，但建议统一尺度
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 保存 scaler 以便后续预测使用
joblib.dump(scaler, "scaler_X.pkl")

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 建立 XGBoost 回归模型（初步设置）
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# 可选：利用 GridSearchCV 调参（这里仅给出示例）
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

grid = GridSearchCV(xgb_model, param_grid, cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

print("最佳参数：", grid.best_params_)
print("最佳交叉验证 R2：", grid.best_score_)

# 使用最优参数建立最终模型
best_model = grid.best_estimator_

# 6. 模型预测与评估
y_pred = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"测试集均方误差 (MSE): {mse:.4f}")
print(f"测试集 R2（决定系数）: {r2:.4f}")

# 7. 绘制预测值与真实值对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="预测值")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color="red", label="理想预测")
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("XGBoost 回归模型：预测值 vs 真实值")
plt.legend()
plt.grid(True)
plt.show()

# 8. 保存模型
joblib.dump(best_model, "xgb_regression_model.pkl")
print("模型和 scaler 已保存！")