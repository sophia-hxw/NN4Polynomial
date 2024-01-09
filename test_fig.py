import torch
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.util import getSaveFileName, getDevice
from models.train import testModel
from utils.visualize import vis_table, get_params
import matplotlib.pyplot as plt

def prepare_fig_data(test_dataset, pred_y):
    x_out = []
    y_out = []
    predy_out = []
    for idx in range(len(test_dataset)):
        x_out.append(round(test_dataset.x[idx].item(),6))
        y_out.append(round(test_dataset.y[idx].item(),6))
        predy_out.append(round(pred_y[idx],6))


    return x_out, y_out, predy_out



def test_model_file(model_name, pt_num):
    # 参数
    file_name = './res_polynomial/fcn/'+ model_name + '.pth'
    device = getDevice()
    print("The device U can Use is: ", device)

    # 数据
    # num_points=100, function_type='polynomial', data_type = 'random', tri_function = 'sin', data_scale = 10,  k=1, seed=None
    test_dataset = CustomDataset(pt_num, "polynomial", "random", None, 10, 5, 31)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # 测试
    # file_name = None, test_loader = None, model = None, device = None, criter = 'MSE', model_type = None
    pred_y, test_loss = testModel(file_name, test_loader, None, device , 'MSE', 'FCN')

    # vis_table(test_dataset, pred_y, test_loss)
    x_out, y_out, predy_out = prepare_fig_data(test_dataset, pred_y)

    # 对x进行排序
    sorted_indices = sorted(range(len(x_out)), key=lambda k: x_out[k])
    x_out_sorted = [x_out[i] for i in sorted_indices]
    y_out_sorted = [y_out[i] for i in sorted_indices]
    predy_out_sorted = [predy_out[i] for i in sorted_indices]

    # 绘制图形并添加图例
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.plot(x_out_sorted, y_out_sorted, color = 'red', marker='*', linestyle='-', label='True y')
    plt.plot(x_out_sorted, predy_out_sorted, color = 'blue', marker='+', linestyle='dashed', label='Predicted y')

    # 添加标题和标签
    plt.title(model_name + '-pt' + str(pt_num))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 保存图像（可以根据需要更改文件名和格式）
    plt.savefig(model_name + '-pt' + str(pt_num) + '.png')

    # 显示图形（可选）
    # plt.show()
    


if __name__ == "__main__":
    model_name = 'FCN_2024-01-09-01-04'
    pt_num = 1000
    test_model_file(model_name, pt_num)   