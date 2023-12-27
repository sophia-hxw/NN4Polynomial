import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的多项式函数 y = 2x^2 - 3x + 1
def polynomial_function(x, n=2, a=1, b=0, c=0):
    return a * x**n - b * x + c

# 生成数据集
torch.manual_seed(42)
x_train = torch.rand(100, 1) * 10
y_train = polynomial_function(x_train)

# 定义 Dense 网络
class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()

        self.fc1 = nn.Linear(1, 64)  # 输入维度为1，输出维度为64
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)  # 输入维度为64，输出维度为64
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)  # 输入维度为64，输出维度为1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 初始化模型、损失函数和优化器
model = DenseNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train)

    # 计算损失
    loss = criterion(outputs, y_train)

    # 反向传播及优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每100次迭代输出一次信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 使用训练好的模型进行预测
with torch.no_grad():
    x_test = torch.rand(1, 1) * 10
    print("x_test shape = ", x_test.shape, "  x_test = ", x_test)
    predicted = model(x_test)
    print("Predicted value:")
    print(predicted.item())
