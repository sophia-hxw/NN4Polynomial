import torch
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, num_points=100, function_type='polynomial', data_type = 'random', tri_function = 'sin', data_scale = 10,  k=1, seed=None):
        self.num_points = num_points
        self.function_type = function_type
        self.data_type = data_type
        self.seed = seed
        self.data_scale = data_scale
        # self.polya = a #多项式最高次项系数
        # self.degree = degree #多项式最高次项次数
        self.k = k #多项式最高次项次数, 三角函数系数
        self.tri_function = tri_function

        # 随机或者一般法生成数据
        if self.data_type == 'general':
            x_val = torch.linspace(-1, 1, self.num_points)
            self.data_generate(x_val)
        elif self.data_type == 'random':
            # 设置随机种子
            torch.manual_seed(seed) if self.seed is not None else None

            # 生成1000个在[0, 1]范围内的随机数
            random_numbers = torch.rand(num_points)

            # 将随机数缩放和平移以获得在[-1, 1]范围内的随机数
            x_val = 2.0 * random_numbers - 1.0

            self.data_generate(x_val)
        else:
            raise ValueError("Invalid function_type. Choose 'general' or 'random'.")

    def data_generate(self, x_val):
        self.x = x_val * self.data_scale
        self.x = torch.round(self.x, decimals = 6) 
        if self.function_type == 'polynomial':
            # self.x = x_val * self.data_scale
            # self.x = torch.round(self.x, decimals=4) 
            self.y = self.x ** self.k
            self.y = torch.round(self.y, decimals = 6) 
        elif self.function_type == 'trigonometric':
            # self.x = x_val * self.data_scale
            # self.x = torch.round(self.x, decimals=4) 
            # self.y = torch.sin(2 * 3.141592654 * self.trik * self.x)
            if self.tri_function == 'sin':
                self.y = torch.sin(self.k * self.x)
            elif self.tri_function == 'cos':
                self.y = torch.cos(self.k * self.x)
            elif self.tri_function == 'tan':
                self.y = torch.tan(self.k * self.x)
            else:
                raise ValueError("Invalid tri_function. Choose 'sin', 'cos' or 'tan'.")
            self.y = torch.round(self.y, decimals = 6) 
        elif self.function_type == 'exponential':
            pass
            # self.y = self.x-self.x0
            # self.y = -self.y.pow(2)/(2*self.k)
            # self.y = torch.exp()
        else:
            raise ValueError("Invalid function_type. Choose 'polynomial' or 'trigonometric'.")

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        return torch.tensor([self.x[index]]), torch.tensor([self.y[index]])

def print_data_parameters():
    pass

def print_model_parameters():
    pass

def print_training_parameters():
    pass
    
def print_testing_parameters():
    pass



