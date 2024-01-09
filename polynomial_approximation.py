import math
import torch
import random
import numpy as np
from models.train import Tester
import torch.nn as nn
import torch.optim as optim

def model_x1(x):
    x = torch.squeeze(x)
    return x.tolist()

def model_x2(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-05-19-34.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)
    
    return pred_y

def model_x3(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-08-15-10.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)

    return pred_y

def model_x4(x):
    file_name = './res_polynomial/fcn/FCN_2024-01-02-03-14.pth'
    device = "cpu"
    print("The device U can Use is: ", device)

    #(model_train = None, model_pth = None, criter = None, device = None) 
    tester = Tester(None, file_name, None, device)

    pred_y = tester.inference_samples(x)

    return pred_y

def right_func(x):
    res = [math.log(1 + val) for val in x]
    return res

    # 定义一个简单的神经网络模型
class LinearSolverNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearSolverNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    x = torch.tensor([0.23, 0.74, 1.12])
    # 使用 torch.unsqueeze 改变张量的形状
    x = torch.unsqueeze(x, dim=1)
    x = torch.unsqueeze(x, dim=2)

    a1 = model_x1(x)
    print("a1 = ", a1)

    a2 = model_x2(x)
    print("a2 = ", a2)

    a3 = model_x3(x)
    print("a3 = ", a3)

    # a4 = model_x4(x)
    # print("a4 = ", a4)

    # 生成系数矩阵
    # M_list = [a1, a2, a3, a4]
    M_list = [a1, a2, a3]
    M = torch.tensor(M_list, dtype=torch.float32)
    M = M.t()
    print("M = ",M)

    # 右端项
    b = [math.sin(val) for val in x]
    b = torch.tensor(b, dtype=torch.float32)
    b = torch.unsqueeze(b, dim=1)
    print("b = ", b)

    # 使用 torch.solve 函数求解线性方程组 Ax = b
    out = torch.linalg.solve(M, b)
    print("Solution torch.linalg.solve x = ")
    print(out)




    
  