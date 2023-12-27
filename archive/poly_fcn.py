import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 定义一个简单的多项式函数，例如 y = 2x^2 + 3x + 1
def true_function(x):
    return 2 * x**2 + 3 * x + 1

# 生成训练数据
def generate_data(num_points):
    x = np.random.uniform(-10, 10, num_points)
    y = true_function(x) + np.random.normal(0, 5, num_points)
    return x, y

class PolynomialDataset(Dataset):
    def __init__(self, num_points):
        self.x, self.y = generate_data(num_points)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.Tensor([self.x[idx]]), torch.Tensor([self.y[idx]])

class FCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FCNModel, self).__init__()
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.fcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcn(x)

# 定义超参数
input_dim = 1
hidden_dims = [64, 64]  # 可根据需要调整隐藏层的维度和层数
output_dim = 1
num_epochs = 1000
lr = 0.001
batch_size = 20

# 创建模型、损失函数和优化器
model = FCNModel(input_dim, hidden_dims, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 准备数据集和数据加载器
train_dataset = PolynomialDataset(num_points=1000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 测试模型
test_x = torch.Tensor(np.linspace(-10, 10, 100)).view(-1, 1)
with torch.no_grad():
    model.eval()
    test_y = model(test_x)

print("test_x = ", test_x)
print("test_y = ", test_y)

# 绘制结果
# plt.scatter(train_dataset.x, train_dataset.y, label='Training Data')
# plt.plot(test_x.numpy(), test_y.numpy(), color='red', label='Fitted Polynomial')
# plt.plot(test_x.numpy(), true_function(test_x.numpy()), '--', color='green', label='True Polynomial')
# plt.legend()
# plt.show()
