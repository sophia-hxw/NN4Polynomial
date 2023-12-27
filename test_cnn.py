"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'

model_type = 'CNN', 'ResNet', 'Dense', 'FCN'
"""

from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import PolynomialCNN
from models.train import testModel, saveModel, Trainer

# TODO: GPU上的训练
# TODO: 模型加深 or 加宽
# 检查 GPU 是否可用
device = getDevice(model_type = 'CNN')
print("The device U can Use is: ", device)

# 准备数据
train_dataset = PolynomialDataset(num_points = 10000, method = 'common', ptimes = 2)
train_loader = DataLoader(train_dataset, batch_size = 20, shuffle=True)

# 定义模型
model = PolynomialCNN(activate = 'LeakyReLU').to(device)


# 创建Trainer实例
trainer = Trainer(model, train_loader, lr = 0.01, criter = 'MSE', checkpoint_dir = './res/checkpoint/', model_type = 'CNN')
trainer.train(num_epochs = 6000, save_interval = 100 )
file_name = getSaveFileName(model_name = 'CNN', path = './res/')
saveModel(model, file_name)

# import torch
# model = torch.load('./res/CNN_2023-12-21-10-41.pth')
# 测试模型
test_dataset = PolynomialDataset(num_points = 15, method = 'random', ptimes = 2)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = 'MSE', model_type = 'cnn')

vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
