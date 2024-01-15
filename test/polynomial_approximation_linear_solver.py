import math
import torch
import random
import numpy as np
from models.train import Tester
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def model_x1(x):
    # x = [(val+1) for val in x]
    x = torch.squeeze(x)
    # return x.tolist()
    x = torch.unsqueeze(x, dim=0)
    return x

def model_x2(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-05-19-34.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)
    pred_y = torch.tensor(pred_y)
    pred_y = torch.unsqueeze(pred_y, dim=0)
    return pred_y

def model_x3(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-08-15-10.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)
    pred_y = torch.tensor(pred_y)
    pred_y = torch.unsqueeze(pred_y, dim=0)
    return pred_y

def model_x4(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-02-03-14.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)
    pred_y = torch.tensor(pred_y)
    pred_y = torch.unsqueeze(pred_y, dim=0)
    return pred_y

def right_func(x):
    # res = [math.log(1 + val) for val in x]
    res = [(math.exp(val)-1) for val in x]
    return res

if __name__ == "__main__":
    # ********** data source **********
    torch.manual_seed(31)
    X = torch.rand(50, 1)  # 生成在[0, 1)范围内的随机数据
    X_source = 2.0 * X - 1.0 #[-1, 1]
    data_scale = 10.0
    X_source = data_scale * X_source
    # print("X = ", X_source)
    X = X_source / data_scale
    y_true = torch.exp(X_source)
    # print("y_true = ", y_true)

    # 将样本数据转换为张量
    X1 = model_x1(X) * data_scale
    # print("X1 = ", X1)

    X2 = model_x2(X) * data_scale * data_scale
    # print("X2 = ", X2)

    X3 = model_x3(X) * data_scale * data_scale * data_scale
    # print("X3 = ", X3)

    X4 = model_x4(X) * data_scale * data_scale * data_scale * data_scale
    # print("X4 = ", X4)

    X_poly = torch.cat((X1, X2, X3, X4), dim = 0)
    X_poly = torch.cat((X1, X2, X3), dim = 0)
    X_poly = X_poly.t()

    out = torch.linalg.solve(X_poly, y_true)
    print("Solution torch.linalg.solve x = ")
    print(out)

        
    pt_num = 1000
    # 绘制结果
    with torch.no_grad():
        X_test = torch.linspace((-1)*data_scale, data_scale, pt_num).view(-1, 1)
        X_in = X_test / data_scale 
        X1_test = model_x1(X_in) * data_scale
        X2_test = model_x2(X_in) * data_scale * data_scale
        X3_test = model_x3(X_in) * data_scale * data_scale * data_scale
        X4_test = model_x4(X_in) * data_scale * data_scale * data_scale * data_scale
        X_poly_test = torch.cat((X1_test, X2_test, X3_test, X4_test), dim = 0)
        # X_poly_test = torch.cat((X1_test, X2_test, X3_test), dim = 0)
        X_poly_test = X_poly_test.t()
        y_pred_test = model(X_poly_test)

    # plt.scatter(X.numpy(), y_true.numpy(),marker='.', color = 'blue', label='True data')
    # plt.plot(X_test.numpy(), y_pred_test.numpy(), marker='+', color='red', label='Fitted curve')
    # plt.legend()
    # plt.show()

    # 绘制图形并添加图例
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.plot(X_source.numpy(), y_true.numpy(), color = 'red', marker='*', linestyle='None', label='True data')
    plt.plot(X_test.numpy(), y_pred_test.numpy(), color = 'blue', marker='+', linestyle='dashed', label='Fitted curve')

    # 添加标题和标签
    plt.title("polynomial approximation of exp(x)")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 保存图像（可以根据需要更改文件名和格式）
    plt.savefig("polynomial approximation of exp(x)-" + str(pt_num) + '.png')

    # 显示图形（可选）
    # plt.show()



    
  