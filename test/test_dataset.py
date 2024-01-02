import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, num_points=100, degree=2, function_type='polynomial', generate_type = 'random', scale = 10, a=1, k=1, seed=None):
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
            self.x_values = x_val * self.scale
            self.x_values = torch.round(self.x_values, decimals=3) 
            self.y_values = self.polya * self.x_values ** self.degree
            self.y_values = torch.round(self.y_values, decimals=3) 
        elif self.function_type == 'trigonometric':
            self.x_values = x_val
            self.x_values = torch.round(self.x_values, decimals=3) 
            self.y_values = torch.sin(2 * 3.141592654 * self.trik * self.x_values)
            self.y_values = torch.round(self.y_values, decimals=3) 
        else:
            raise ValueError("Invalid function_type. Choose 'polynomial' or 'trigonometric'.")

    def __len__(self):
        return self.num_points

    def __getitem__(self, index):
        return self.x_values[index], self.y_values[index]

if __name__ == "__main__":
    # 生成二次多项式数据
    poly_dataset = CustomDataset(num_points=10, degree=2, function_type='polynomial', generate_type = 'random', scale = 10, a=1, k=1, seed=42)

    # 生成三角函数数据
    trig_dataset = CustomDataset(num_points=10, function_type='trigonometric', generate_type = 'general', scale = 10, a=1, k=1, seed=42)

    # 打印前几个数据点
    print("Polynomial Data:")
    for x, y in poly_dataset:
        print(f"{x.item():.4f}: {y.item():.4f}")
        # print(f"{round(x.item(),3)}: {round(y.item(),3)}")

    print("\nTrigonometric Data:")
    for x, y in trig_dataset:
        print(f"{x.item():.4f}: {y.item():.4f}")
        # print(f"{round(x.item(),3)}: {round(y.item(),3)}")
