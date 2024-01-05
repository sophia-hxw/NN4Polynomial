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

device = getDevice()
print("The device U can Use is: ", device)

dtparams, traparams, mdlparams, tstparams = get_params('./configs/trigo_fcn.json')

# ****************************** DATA ****************************************
# num_points=100, function_type='polynomial', data_type = 'random', tri_function = 'sin', data_scale = 10,  k=1, seed=None
train_dataset = CustomDataset(dtparams['num_points'], dtparams['function_type'], dtparams['data_type'], dtparams['tri_function'], dtparams['data_scale'], dtparams['k'], None)
# print("Polynomial Data:")
# for x, y in train_dataset:
#     print(f"{x.item():.6f}: {y.item():.6f}")
train_loader = DataLoader(train_dataset, batch_size = dtparams['batch_size'], shuffle = True)
# num_points=100, function_type='polynomial', data_type = 'random', tri_function = 'sin', data_scale = 10,  k=1, seed=None
test_dataset = CustomDataset(tstparams['num_points'], dtparams['function_type'], tstparams['data_type'], dtparams['tri_function'], tstparams['data_scale'], dtparams['k'], tstparams['seed'])
test_loader = DataLoader(test_dataset, tstparams['batch_size'], shuffle = False)

# ****************************** MODEL ****************************************
model = FCNModel(mdlparams['layer_dims'], mdlparams['activation'])

# ****************************** TRAIN ****************************************
file_name = getSaveFileName(mdlparams['model_type'], traparams['res_dir'])
trainer = Trainer(model, train_loader, test_loader, traparams['learning_rate'], traparams['criter'], traparams['res_dir']), mdlparams['model_type'])
pred_y, test_loss = trainer.train(traparams['num_epochs'], traparams['save_interval'], file_name, device )

# ****************************** VISUALIZATION ****************************************
vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
