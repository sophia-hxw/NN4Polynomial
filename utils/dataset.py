import torch
import numpy as np
from torch.utils.data import Dataset

def polynomial_function(x, n=2, a=1, b=0, c=0):
    return a * x**n - b * x + c

def random_generate_data(num_points, ptimes = 2):
    # np.random.seed(42)
    # x = np.random.uniform(-10, 10, num_points)
    # # np.random.normal(0, 5, num_points)
    # # 生成 num_points 个符合均值为0，标准差为5的概率密度随机数
    # # y = polynomial_function(x) + np.random.normal(0, 5, num_points)
    # y = polynomial_function(x)

    torch.manual_seed(42)

    # 生成1000个在[0, 1]范围内的随机数
    random_numbers = torch.rand(num_points)

    # 将随机数缩放和平移以获得在[-1, 1]范围内的随机数
    random_numbers_scaled = 2.0 * random_numbers - 1.0

    x = random_numbers_scaled * 10.0
    y = polynomial_function(x, ptimes)

    return x, y

def common_generate_data(num_points, ptimes = 2):
    x = torch.linspace(-1, 1, num_points) * 10.0
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

