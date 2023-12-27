"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""

from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import ResidualBlock, ResNet
from models.train import trainModel, testModel, saveModel

# TODO: 模型测试
# 检查 GPU 是否可用
device = getDevice(model_type = 'resnet')
print("The device U can Use is: ", device)

# 准备数据
train_dataset = PolynomialDataset(num_points = 10000, method = 'common')
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 定义模型
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)

# 训练模型
trainModel(train_loader, model, device, num_epochs = 6000, criter = 'MSE')
file_name = getSaveFileName(model_name = 'ResNet', path = './res/')
saveModel(model, file_name)

# import torch
# model = torch.load('./res/Dense_2023-12-20-20.pth')
# 测试模型
test_dataset = PolynomialDataset(num_points = 15, method = 'random')
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = 'MSE')

vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
