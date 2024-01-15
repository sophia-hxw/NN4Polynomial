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

def gen_coe_mat(X, data_scale):
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

    return  X_poly.t()

def gen_data_random(data_scale = 1.0, pt_num = 16, seed = 30):
    torch.manual_seed(seed)
    X = torch.rand(pt_num, 1)  # 生成在[0, 1)范围内的随机数据
    X_source = 2.0 * X - 1.0 #[-1, 1]
    X_source = data_scale * X_source #[-data_scale, data_scale]
    # print("X = ", X_source)
    return  X_source

def least_square():
    pass

# 定义多项式模型
class PolynomialModel(nn.Module):
    def __init__(self, degree):
        super(PolynomialModel, self).__init__()
        self.poly = nn.Linear(degree, 1)

    def forward(self, x):
        # return torch.exp(self.poly(x))
        # print("self.poly(x) = ", self.poly(x))
        return self.poly(x)

if __name__ == "__main__":
    # ********** data source **********
    data_scale = 10.0
    X_source = gen_data_random(data_scale, 16, 30)

    y_true = torch.exp(X_source)
    # print("y_true = ", y_true)
    X_in_model = X_source / data_scale
 
    # 初始化模型、损失函数和优化器
    degree = 4  # 多项式的次数
    model = PolynomialModel(degree)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.005)

    # 训练模型
    num_epochs = 100000 
    for epoch in range(num_epochs):
        X_poly = gen_coe_mat(X_in_model, data_scale)
        # print("X_poly = ", X_poly)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f'{name}: {param.data}')
        # 前向传播
        y_pred = model(X_poly)
        # print("y_pred = ", y_pred)

        # 计算损失
        loss = criterion(y_pred, y_true)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.8f}')
            # print("y_pred = ", y_pred)
        
    pt_num = 10
    # 绘制结果
    with torch.no_grad():
        X_test = torch.linspace((-1)*data_scale, data_scale, pt_num).view(-1, 1)
        # print("X_test = ", X_test)
        y_label_test = torch.exp(X_test)
        # print("y_label_test = ", y_label_test)
        X_in = X_test / data_scale 
        X1_test = model_x1(X_in) * data_scale
        X2_test = model_x2(X_in) * data_scale * data_scale
        X3_test = model_x3(X_in) * data_scale * data_scale * data_scale
        X4_test = model_x4(X_in) * data_scale * data_scale * data_scale * data_scale
        X_poly_test = torch.cat((X1_test, X2_test, X3_test, X4_test), dim = 0)
        X_poly_test = X_poly_test.t()
        y_pred_test = model(X_poly_test)
        # print("y_pred_test = ", y_pred_test)   
    
    
    # 绘制图形并添加图例
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.plot(X_source, y_true, color = 'red', marker='*', linestyle='None', label='True data')
    plt.plot(X_test, y_pred_test, color = 'blue', marker='.', linestyle='None', label='Fitted curve')

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



    
  