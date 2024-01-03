"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.visualize import vis_table, save_json, readjson
from utils.util import getSaveFileName, getDevice
from models.networks import FCNModel
from models.train import testModel, saveModel, Trainer

# TODO: 网络测试 【完成，结论：可用】
# TODO: 检查 GPU 是否可用
device = getDevice()
print("The device U can Use is: ", device)

params = readjson('./configs/params_fcn.json')
dtparams = params['data']
traparams = params['training']
mdlparams = params['model']
tstparams = params['testing']

# ****************************** DATA and MODEL ****************************************
# init(num_points = 100, function_type='polynomial', data_type = 'random', tri_function = 'sin', degree=2, scale = 10, a=1, k=1, seed=None)
train_dataset = CustomDataset(dtparams['num_points'], dtparams['function_type'], dtparams['data_type'], dtparams['tri_function'], None, dtparams['data_scale'], None, dtparams['tri_k'], None)
train_loader = DataLoader(train_dataset, batch_size = dtparams['batch_size'], shuffle = True)

model = FCNModel(mdlparams['input_dim'], mdlparams['hidden_dims'], mdlparams['output_dim'])

# ****************************** TRAIN ****************************************
# init(model, train_loader, lr=0.01, criter = 'MSE', checkpoint_dir = './cache/', model_type = None)
trainer = Trainer(model, train_loader, traparams['learning_rate'], traparams['criter'], traparams['checkpoint'], mdlparams['model_type'])
# train(num_epochs = 5000, save_interval = 1000, device = None)
trainer.train(num_epochs = traparams['num_epochs'], save_interval = traparams['save_interval'] )
# getSaveFileName(model_name = 'FCN', path = './res/')
file_name = getSaveFileName(mdlparams['model_type'], traparams['res_dir'])
saveModel(model, file_name)

# ****************************** TEST ****************************************
# import torch
# model = torch.load('./res/FCN_2024-01-03-00-11.pth')
# init(num_points = 100, function_type='polynomial', data_type = 'random', tri_function = 'sin', degree=2, scale = 10, a=1, k=1, seed=None)
test_dataset = CustomDataset(tstparams['num_points'], dtparams['function_type'], tstparams['data_type'], dtparams['tri_function'], None, tstparams['data_scale'], None, dtparams['tri_k'], tstparams['seed'])
test_loader = DataLoader(test_dataset, tstparams['batch_size'], shuffle = False)
pred_y, test_loss = testModel(test_loader, model, device, criter = traparams['criter'])

# ****************************** VISUALIZATION ****************************************
vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
