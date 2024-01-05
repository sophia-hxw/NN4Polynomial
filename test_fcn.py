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

# TODO: 网络测试 【完成，结论：可用】
# TODO: 检查 GPU 是否可用【完成，结论：可用】
# TODO: 检查数据集是否正确
device = getDevice()
print("The device U can Use is: ", device)

dtparams, traparams, mdlparams, tstparams = get_params('./configs/params_fcn.json')

# ****************************** DATA ****************************************
train_dataset = CustomDataset(dtparams['num_points'], dtparams['function_type'], dtparams['data_type'], dtparams['tri_function'], None, dtparams['data_scale'], None, dtparams['tri_k'], None)
# print("Polynomial Data:")
# for x, y in train_dataset:
#     print(f"{x.item():.6f}: {y.item():.6f}")
train_loader = DataLoader(train_dataset, batch_size = dtparams['batch_size'], shuffle = True)

test_dataset = CustomDataset(tstparams['num_points'], dtparams['function_type'], tstparams['data_type'], dtparams['tri_function'], None, tstparams['data_scale'], None, dtparams['tri_k'], tstparams['seed'])
test_loader = DataLoader(test_dataset, tstparams['batch_size'], shuffle = False)

# ****************************** MODEL ****************************************
model = FCNModel(mdlparams['input_dim'], mdlparams['hidden_dims'], mdlparams['output_dim'], mdlparams['activation'])

# ****************************** TRAIN ****************************************
file_name = getSaveFileName(mdlparams['model_type'], traparams['res_dir'])
trainer = Trainer(model, train_loader, test_loader, traparams['learning_rate'], traparams['criter'], traparams['checkpoint'], mdlparams['model_type'])
trainer.train(traparams['num_epochs'], traparams['save_interval'], file_name, device )

# ****************************** VISUALIZATION ****************************************
vis_table(test_dataset, pred_y, test_loss)
save_json(file_name, test_dataset, pred_y, test_loss)
