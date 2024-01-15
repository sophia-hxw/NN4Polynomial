from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from utils.visualize import vis_table, save_json
from utils.util import getSaveFileName, getDevice
from models.networks import PolynomialCNN
from models.train import testModel, saveModel, Trainer

device = getDevice(model_type = 'CNN')
print("The device U can Use is: ", device)

dtparams, traparams, mdlparams, tstparams = get_params('./configs/trigo_fcn.json')

# ****************************** DATA ****************************************
train_dataset = CustomDataset(dtparams['num_points'], dtparams['function_type'], dtparams['data_type'], dtparams['tri_function'], dtparams['data_scale'], dtparams['k'], None)
train_loader = DataLoader(train_dataset, batch_size = dtparams['batch_size'], shuffle = True)
test_dataset = CustomDataset(tstparams['num_points'], dtparams['function_type'], tstparams['data_type'], dtparams['tri_function'], tstparams['data_scale'], dtparams['k'], tstparams['seed'])
test_loader = DataLoader(test_dataset, tstparams['batch_size'], shuffle = False)

# ****************************** MODEL ****************************************
model = PolynomialCNN(activate = 'LeakyReLU').to(device)
model = CNN(mdlparams['layer_dims'], mdlparams['activation'])



# 创建Trainer实例
trainer = Trainer(model, train_loader, lr = 0.00004, criter = 'MSE', checkpoint_dir = './res/checkpoint/', model_type = 'CNN')
trainer.train(num_epochs = 15000, save_interval = 5000 )
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
