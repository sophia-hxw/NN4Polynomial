"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""
from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import FCNModel
from models.train import testModel, saveModel, Trainer

# TODO: 网络测试 【完成，结论：可用】
# TODO: 检查 GPU 是否可用
device = getDevice()
print("The device U can Use is: ", device)

# 准备数据
train_dataset = PolynomialDataset(num_points = 60000, method = 'common', ptimes = 4)
train_loader = DataLoader(train_dataset, batch_size = 50, shuffle=True)

# 定义超参数
input_dim = 1
hidden_dims = [32, 64, 128, 256, 256, 128, 64, 32]  # 可根据需要调整隐藏层的维度和层数
output_dim = 1
model = FCNModel(input_dim, hidden_dims, output_dim)

# 创建Trainer实例
trainer = Trainer(model, train_loader, lr = 0.0004, criter = 'MSE', checkpoint_dir = './res/checkpoint/', model_type = 'FCN')
trainer.train(num_epochs = 40000, save_interval = 5000 )
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
