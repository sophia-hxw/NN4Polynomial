import torch
import numpy as np
from torch.utils.data import Dataset

def polynomial_function(x, n=2, a=1, b=0, c=0):
    return a * x**n - b * x + c

def trigonometric_functions(x, k = 1):
    return sin(k * x)

def random_generate_data(num_points, ptimes = 2, scale = 10.0):
    torch.manual_seed(42)

    # 生成1000个在[0, 1]范围内的随机数
    random_numbers = torch.rand(num_points)

    # 将随机数缩放和平移以获得在[-1, 1]范围内的随机数
    random_numbers_scaled = 2.0 * random_numbers - 1.0

    x = random_numbers_scaled * scale
    y = polynomial_function(x, ptimes)

    return x, y

def common_generate_data(num_points, ptimes = 2, scale = 10.0):
    x = torch.linspace(-1, 1, num_points) * scale
    y = polynomial_function(x, ptimes)
    return x, y

def preprocess_data(x, y):
    x = torch.from_numpy(x).float().view(-1, 1)
    y = torch.from_numpy(y).float().view(-1, 1)
    # print("x shape =", x.shape)
    # print("y shape =", y.shape)
    return x, y

class PolynomialDataset(Dataset):
    def __init__(self, num_points=1000, method = 'random', ptimes = 2):
        if method == 'random':
            self.x, self.y = random_generate_data(num_points, ptimes)
        elif method == 'common':
            self.x, self.y = common_generate_data(num_points, ptimes)
        else:
            print("The default data generation method is random generation method.")

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor([self.x[idx]]), torch.tensor([self.y[idx]])

class TrigonometricDataset(Dataset):
    def __init__(self, num_points=1000, method = 'random', ptimes = 2):
        if method == 'random':
            self.x, self.y = random_generate_data(num_points, ptimes)
        elif method == 'common':
            self.x, self.y = common_generate_data(num_points, ptimes)
        else:
            print("The default data generation method is random generation method.")

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.tensor([self.x[idx]]), torch.tensor([self.y[idx]])


class CustomDataset(Dataset):
    def __init__(self, num_points=100, function_type='polynomial', generate_type = 'random', degree=2, scale = 10, a=1, k=1, seed=None):
        self.num_points = num_points
        self.function_type = function_type
        self.generate_type = generate_type
        self.seed = seed
        self.scale = scale
        self.polya = a #多项式最高次项系数
        self.degree = degree #多项式最高次项次数
        self.trik = k #三角函数系数

        # 随机或者一般法生成数据
        if self.generate_type == 'general':
            x_val = torch.linspace(-1, 1, self.num_points)
            self.data_generate(x_val)
        elif self.generate_type == 'random':
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
        if self.function_type == 'polynomial':
            self.x = x_val * self.scale
            self.x = torch.round(self.x, decimals=3) 
            self.y = self.polya * self.x ** self.degree
            self.y = torch.round(self.y, decimals=3) 
        elif self.function_type == 'trigonometric':
            self.x = x_val * self.scale
            self.x = torch.round(self.x, decimals=3) 
            # self.y = torch.sin(2 * 3.141592654 * self.trik * self.x)
            self.y = torch.sin(self.trik * self.x)
            self.y = torch.round(self.y, decimals=3) 
        else:
            raise ValueError("Invalid function_type. Choose 'polynomial' or 'trigonometric'.")

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        return torch.tensor([self.x[index]]), torch.tensor([self.y[index]])