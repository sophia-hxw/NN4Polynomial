import torch
import json
import numpy as np
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的多项式函数 y = 2x^2 - 3x + 1
def polynomial_function(x, n=2, a=1, b=0, c=0):
    return a * x**n - b * x + c

# 生成数据集
# torch.manual_seed(21)
# x_train = torch.rand(10, 1) * 10
# y_train = polynomial_function(x_train, 3)
x_train = np.linspace(0,1,2000)*10
x_train = torch.tensor(x_train, dtype=torch.float32).reshape(-1,1)
y_train = polynomial_function(x_train, 3)

# 定义神经网络模型
class PolynomialCNN(nn.Module):
    def __init__(self):
        super(PolynomialCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 10, kernel_size=1)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(10, 20, kernel_size=1)
        self.conv3 = nn.Conv1d(20, 40, kernel_size=1)
        self.conv4 = nn.Conv1d(40, 80, kernel_size=1)
        self.conv5 = nn.Conv1d(80, 1, kernel_size=1)

    def forward(self, x):
        # print("original x = ", x.shape)
        x = self.conv1(x)
        # print("conv1 x = ", x.shape)
        x = self.relu(x)
        # print("relu1 x = ", x.shape)
        x = self.conv2(x)
        # print("conv2 x = ", x.shape)
        x = self.relu(x)
        # print("relu2 x = ", x.shape)
        x = self.conv3(x)
        # print("conv3 x = ", x.shape)
        x = self.relu(x)
        # print("relu3 x = ", x.shape)
        x = self.conv4(x)
        # print("conv4 x = ", x.shape)
        x = self.relu(x)
        # print("relu4 x = ", x.shape)
        x = self.conv5(x)
        # print("conv5 x = ", x.shape)
        return x

# 初始化模型、损失函数和优化器
model = PolynomialCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 将数据整理成 CNN 输入的形状
x_train_input = x_train.unsqueeze(1).permute(0, 2, 1)

# 训练模型
cell_epochs = 2000
test_epochs = 4
for test_epoch in range(test_epochs):
    num_epochs = (test_epoch+1) * cell_epochs
    test_model_name = 'model_weights_' + str(test_epoch) 
    test_model_res = {'ModalName': test_model_name}
    for epoch in range(num_epochs):
        # 前向传播
        outputs = model(x_train_input)

        # 计算损失
        loss = criterion(outputs, y_train.unsqueeze(1))

        # 反向传播及优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每500次迭代输出一次信息
        if (epoch + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        
            res_key = 'Epoch ' + str(epoch+1) + '/' + str(num_epochs)
            # res_value = 'Loss ' + str("{:.4f}".format(loss.item()))
            res_value = 'Loss ' + str(round(loss.item(), 4))
            test_model_res.update({res_key: res_value})

    # 使用训练好的模型进行预测
    with torch.no_grad():
        # x_test = torch.rand(1, 1) * 10
        x_test = torch.tensor([[9.7399]]) # 9.7399*9.7399=94.86565201
        x_test_input = x_test.unsqueeze(1).permute(0, 2, 1)
        print("input value:")
        print(x_test_input)
        predicted = model(x_test_input)
        print("Predicted value:")
        print(predicted.item())

    # 保存模型
    torch.save(model, test_model_name + '.pth')

    #
    file_name = test_model_name + '.json'
    with open(file_name, 'w') as json_file:
        json.dump(test_model_res, json_file)