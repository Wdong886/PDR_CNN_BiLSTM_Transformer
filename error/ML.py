import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pandas as pd

# 读取数据
file_path = r"D:\博士\IMU定位\final\PersonDeadReckoning\data\1\1_combined_excel.xlsx"
df = pd.read_excel(file_path, sheet_name='Sheet1', engine='openpyxl')
X = df.iloc[:, :93].values
y = df.iloc[:, -2].values

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)


# 定义深度学习模型
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# 初始化模型、损失函数和优化器
model = RegressionModel(input_dim=X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 500
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor)
            r2 = r2_score(y_test, y_pred.numpy())
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, R2: {r2:.3f}")

# 最终评估
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    final_r2 = r2_score(y_test, y_pred.numpy())
print(f"\nFinal R2: {final_r2:.3f}")