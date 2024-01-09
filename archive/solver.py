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
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 4)

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

# **************线性方程组******************
# import math
# import torch
# import random
# import numpy as np
# from models.train import Tester
# import torch.nn as nn
# import torch.optim as optim

a1 = [1.0, 1.0, 3.0, 2.0]
a2 = [1.0, -1.0, 1.0, 1.0]
a3 = [1.0, 2.0, -6.0, -1.0]
a4 = [1.0, 1.0, 5.0, -1.0]
b = [10.0, 9.0, 7.0, -3.0]
# 生成系数矩阵
M_list = [a1, a2, a3, a4]
M = torch.tensor(M_list, dtype=torch.float32)
M = M.t()
print("M = ",M)

# 右端项
# b = [math.log(1 + val) for val in x]
b = torch.tensor(b, dtype=torch.float32)
b = torch.unsqueeze(b, dim=1)
print("b = ", b)

# 使用 torch.solve 函数求解线性方程组 Ax = b
out = torch.linalg.solve(M, b)
print("Solution torch.linalg.solve x = ")
print(out)
# solution = [1.0, 2.0, 3.0, 4.0]



# # 定义系数矩阵 A 和右侧向量 b
# M = torch.tensor([[2.0, -1.0], [1.0, 3.0]])
# b = torch.tensor([[1.0], [4.0]])

# 将数据转换为 PyTorch 可以处理的格式
M_tensor = torch.FloatTensor(M)
b_tensor = torch.FloatTensor(b)

# 定义神经网络模型
input_size = 4  # 系数矩阵 A 的列数
output_size = 1  # 解向量 x 的维度
model = LinearSolverNN(input_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.0005)

# 训练模型
epochs = 10000
for epoch in range(epochs):
    # 前向传播
    output = model(M_tensor)
    # print("output shape = ", output.shape)
    # print("b_tensor shape= ", b_tensor.shape)
    # 计算损失
    loss = criterion(output, b_tensor)
    print("loss = ", loss.item())
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 获取模型的权重作为解
solution = model.linear.weight.detach().numpy()

print("Solution NN x = ")
print(solution)
