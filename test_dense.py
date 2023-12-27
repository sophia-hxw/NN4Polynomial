"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""

from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import DenseNet
from models.train import testModel, saveModel, Trainer

# 检查 GPU 是否可用
device = getDevice()
print("The device U can Use is: ", device)

# 准备数据
# TODO: 训练三次多项式，精度改进方向：网络结构调整+训练数据和迭代次数增加
train_dataset = PolynomialDataset(num_points = 10000, method = 'common', ptimes = 5)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

# 定义模型
model = DenseNet(activate = 'LeakyReLU').to(device)

# 创建Trainer实例
trainer = Trainer(model, train_loader, lr = 0.0001, criter = 'MSE', checkpoint_dir = './res/checkpoint/', model_type = 'Dense')
trainer.train(num_epochs = 8000, save_interval = 100)
file_name = getSaveFileName(model_name = 'Dense', path = './res/')
saveModel(model, file_name)

# import torch
# model = torch.load('./res/Dense_2023-12-22-12-15.pth')
# 测试模型
test_dataset = PolynomialDataset(num_points = 15, method = 'random', ptimes = 5)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = 'MSE')

vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
