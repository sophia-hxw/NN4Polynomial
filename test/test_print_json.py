import torch
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.util import getSaveFileName, getDevice
from models.train import testModel
from utils.visualize import vis_table

# 参数
file_name = './res_trigonometric/fcn/FCN_2024-01-03-20-16.pth'
device = getDevice()
print("The device U can Use is: ", device)

# 数据
test_dataset = CustomDataset(20, "trigonometric", "random", "sin", None, 10, None, 10, 42)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

# 测试
# testModel(file_name = None, test_loader = None, model = None, device = None, criter = 'MSE', model_type = None)
pred_y, test_loss = testModel(file_name, test_loader, model =  None, device = device , criter = 'MSE', model_type = 'FCN')

vis_table(test_dataset, pred_y, test_loss)
# save_json(file_name, test_dataset, pred_y, test_loss)