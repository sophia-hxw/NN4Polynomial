"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""
from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset, CustomDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import FCNModel
from models.train import testModel, saveModel, Trainer

# TODO: 网络测试 【完成，结论：可用】
# TODO: 检查 GPU 是否可用
device = getDevice()
print("The device U can Use is: ", device)

# 准备数据
# 生成二次多项式数据
# train_dataset = CustomDataset(num_points=1000, function_type='polynomial', generate_type = 'general', degree=2, scale = 10, a=1)
# train_loader = DataLoader(train_dataset, batch_size = 50, shuffle=True)

# 生成三角函数数据
train_dataset = CustomDataset(num_points = 20000, function_type='trigonometric', generate_type = 'general', scale = 10, k=1)
train_loader = DataLoader(train_dataset, batch_size = 20, shuffle=True)

# 定义超参数
input_dim = 1
hidden_dims = [16, 32, 64, 128, 256, 512, 512, 256, 128, 64, 32, 16]  # 可根据需要调整隐藏层的维度和层数
output_dim = 1
model = FCNModel(input_dim, hidden_dims, output_dim)

# 创建Trainer实例
trainer = Trainer(model, train_loader, lr = 0.001, criter = 'MSE', checkpoint_dir = './res/checkpoint/', model_type = 'FCN')
trainer.train(num_epochs = 8000, save_interval = 1000 )
file_name = getSaveFileName(model_name = 'FCN', path = './res/')
saveModel(model, file_name)

# import torch
# model = torch.load('./res/FCN_2024-01-03-00-11.pth')
# 测试模型
# 生成二次多项式数据
test_dataset = CustomDataset(num_points=15, function_type='trigonometric', generate_type = 'random', scale = 10, k = 1, seed = 42)
# test_dataset = PolynomialDataset(num_points = 15, method = 'random', ptimes = 5)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = 'MSE')

vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
