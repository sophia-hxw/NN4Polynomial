"""
activation: 'ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid', 'Mish'
lossfunction: 'MSE', 'CrossEntropy', 'L1'
"""
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.visualize import vis_table, save_json, get_params
from utils.util import getSaveFileName, getDevice
from models.networks import FCNModel
from models.train import testModel, Trainer

# TODO: [x]网络测试 【完成，结论：可用】
# TODO: 检查 GPU 是否可用【完成，结论：可用】
# TODO: 检查数据集是否正确
device = getDevice()
print("The device U can Use is: ", device)

# return params['data'], params['training'], params['model'], params['testing']
dtparams, traparams, mdlparams, tstparams = get_params('./configs/params_fcn.json')

# ****************************** DATA and MODEL ****************************************
# init(num_points = 100, function_type='polynomial', data_type = 'random', tri_function = 'sin', degree=2, scale = 10, a=1, k=1, seed=None)
train_dataset = CustomDataset(dtparams['num_points'], dtparams['function_type'], dtparams['data_type'], dtparams['tri_function'], None, dtparams['data_scale'], None, dtparams['tri_k'], None)
train_loader = DataLoader(train_dataset, batch_size = dtparams['batch_size'], shuffle = True)
# init(input_dim, hidden_dims, output_dim, activate)
model = FCNModel(mdlparams['input_dim'], mdlparams['hidden_dims'], mdlparams['output_dim'], mdlparams['activation'])

# ****************************** TRAIN ****************************************
# getSaveFileName(model_name = 'FCN', path = './res/')
file_name = getSaveFileName(mdlparams['model_type'], traparams['res_dir'])
# init(model, train_loader, lr=0.01, criter = 'MSE', checkpoint_dir = './cache/', model_type = None)
trainer = Trainer(model, train_loader, traparams['learning_rate'], traparams['criter'], traparams['checkpoint'], mdlparams['model_type'])
# train(num_epochs = 5000, save_interval = 1000, file_name = None, device = None)
trainer.train(traparams['num_epochs'], traparams['save_interval'], file_name, device )

# ****************************** TEST ****************************************
# init(num_points = 100, function_type='polynomial', data_type = 'random', tri_function = 'sin', degree=2, scale = 10, a=1, k=1, seed=None)
test_dataset = CustomDataset(tstparams['num_points'], dtparams['function_type'], tstparams['data_type'], dtparams['tri_function'], None, tstparams['data_scale'], None, dtparams['tri_k'], tstparams['seed'])
test_loader = DataLoader(test_dataset, tstparams['batch_size'], shuffle = False)
# (file_name = None, test_loader = None, model = None, device = None, criter = 'MSE', model_type = None)
pred_y, test_loss = testModel(None, test_loader, model, device, traparams['criter'], mdlparams['model_type'])

# ****************************** VISUALIZATION ****************************************
vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
