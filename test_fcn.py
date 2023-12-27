"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""
from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import FCNModel
from models.train import trainModel, testModel, saveModel

# TODO: 网络测试
# 检查 GPU 是否可用
device = getDevice()
print("The device U can Use is: ", device)

# 准备数据
train_dataset = PolynomialDataset(num_points = 20000, method = 'common', ptimes = 4)
train_loader = DataLoader(train_dataset, batch_size = 5, shuffle=True)

# 定义超参数
input_dim = 1
hidden_dims = [64, 128, 256, 128, 64]  # 可根据需要调整隐藏层的维度和层数
output_dim = 1
model = FCNModel(input_dim, hidden_dims, output_dim)

# 训练模型
trainModel(train_loader, model, device, num_epochs = 10000, criter = 'MSE', lr=0.00005, model_type = 'fcn')
file_name = getSaveFileName(model_name = 'FCN', path = './res/')
saveModel(model, file_name)

# import torch
# model = torch.load('./res/Dense_2023-12-20-20.pth')
# 测试模型
test_dataset = PolynomialDataset(num_points = 15, method = 'random', ptimes = 4)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = 'MSE')

vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
