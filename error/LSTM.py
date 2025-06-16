import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBRegressor
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# 1. 读取 Excel 数据
file_path = r"../data\1\1_combined_excel.xlsx"  # 修改为实际路径
df = pd.read_excel(file_path, sheet_name="Sheet1", engine="openpyxl")

# 2. 数据预处理
# 将所有数据转为数值型，处理缺失值
df = df.apply(pd.to_numeric, errors="coerce")
df = df.dropna()  # 删除含缺失值的行

# 3. 异常值处理（使用IQR方法）
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# 对目标列进行异常值处理
target_column = df.columns[-2]  # 假设目标列是倒数第二列
df = remove_outliers(df, target_column)

# 4. 分离特征与目标
# 特征：前92列
X = df.iloc[:, :90].values
# 目标：倒数第二列
y = df.iloc[:, -2].values

# 5. 特征选择（基于XGBoost特征重要性）
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X, y)
importances = xgb_model.feature_importances_

# 选择重要性大于中位数的特征
threshold = np.median(importances)
selected_features = np.where(importances > threshold)[0]
X_selected = X[:, selected_features]

# 6. 特征交互（添加多项式特征）
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_interacted = poly.fit_transform(X_selected)

# 7. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_interacted)

# 8. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 9. 构建深度学习模型（MLP）
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # 输出层
])

# 编译模型
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# 早停法防止过拟合
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 训练模型
history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# 10. 评估模型
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"测试集均方误差 (MSE): {mse:.4f}")
print(f"测试集 R2（决定系数）: {r2:.4f}")

# 11. 绘制预测值与真实值对比图
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color="blue", label="预测值")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--', color="red", label="理想预测")
plt.xlabel("真实值")
plt.ylabel("预测值")
plt.title("MLP 回归模型：预测值 vs 真实值")
plt.legend()
plt.grid(True)
plt.show()

# 12. 保存模型
model.save("mlp_regression_model.h5")
print("模型已保存！")