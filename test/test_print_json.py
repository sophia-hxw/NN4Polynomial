import torch
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.util import getSaveFileName, getDevice
from models.train import testModel
from utils.visualize import vis_table, get_params

def test_model_file():
    # 参数
    file_name = './res_polynomial/fcn/FCN_2024-01-05-21-27.pth'
    device = getDevice()
    print("The device U can Use is: ", device)

    # 数据
    # num_points=100, function_type='polynomial', data_type = 'random', tri_function = 'sin', data_scale = 10,  k=1, seed=None
    test_dataset = CustomDataset(20, "polynomial", "random", None, 10, 2, 31)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # 测试
    # file_name = None, test_loader = None, model = None, device = None, criter = 'MSE', model_type = None
    pred_y, test_loss = testModel(file_name, test_loader, None, device , 'MSE', 'FCN')

    vis_table(test_dataset, pred_y, test_loss)

def test_json_null():
    dtparams, traparams, mdlparams, tstparams = get_params('./configs/params_fcn.json')

    print("all data parameters: ", dtparams)
    print("test of null in json: ", dtparams["test_key"])

if __name__ == "__main__":
    test_model_file()    
