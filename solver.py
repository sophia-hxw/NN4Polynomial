import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的非线性三元方程组
def nonlinear_equations(x, y, z):
    eq1 = x**2 + y - 4
    eq2 = x - y**2 + z - 1
    eq3 = x + y + z**2 - 9
    return eq1, eq2, eq3

# 生成数据集
torch.manual_seed(42)
data_size = 100
x_train = torch.rand(data_size, 1) * 10
y_train = torch.rand(data_size, 1) * 10
z_train = torch.rand(data_size, 1) * 10
eq1, eq2, eq3 = nonlinear_equations(x_train, y_train, z_train)

# 将方程组的结果合并成一个张量
equations_result = torch.cat([eq1, eq2, eq3], dim=1)

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(equations_result)

    # 计算损失
    loss = criterion(outputs, torch.zeros_like(outputs))

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
    y_test = torch.rand(1, 1) * 10
    z_test = torch.rand(1, 1) * 10
    test_input = torch.cat([x_test, y_test, z_test], dim=1)
    
    predicted = model(test_input)
    print("Predictions:")
    print(predicted)
