import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import PolynomialDataset
from models.networks import FCNModel

# class SimpleModel(nn.Module):
#     def __init__(self):
#         super(SimpleModel, self).__init__()
#         self.fc = nn.Linear(10, 1)

#     def forward(self, x):
#         return self.fc(x)

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, checkpoint_dir='./checkpoint'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, num_epochs=5000, save_interval=1000):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print("checkpoint_path: ", checkpoint_path)

        if os.path.exists(checkpoint_path):
            # 如果存在checkpoint，加载模型参数
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint.get('model_state_dict', self.model.state_dict()))
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', self.optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', 0)
            best_loss = checkpoint.get('loss', float('inf'))
        else:
            start_epoch = 0
            best_loss = float('inf')

        for epoch in range(start_epoch, start_epoch + num_epochs):
            # self.model.train()
            for inputs, targets in self.train_loader:
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

            # 保存checkpoint
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, checkpoint_path)

import os
import re

def find_max_pth_with_string(folder_path='.', search_string=''):
    pth_files = [file for file in os.listdir(folder_path) if file.endswith('.pth') and search_string in file]

    if not pth_files:
        return None  # If no matching files are found, return None

    max_number = float('-inf')  # Initialize the maximum number to negative infinity
    max_pth_file = None

    for pth_file in pth_files:
        match = re.search(r'_(\d+)_', pth_file)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                max_pth_file = pth_file

    return max_pth_file

if __name__ == '__main__':
    current_folder = './cache'  # Current folder
    given_string = 'cnn'  # Given string to match in the filename
    max_pth_file = find_max_pth_with_string(current_folder, given_string)

    if max_pth_file:
        print(f"The .pth file with the maximum number and '{given_string}' in the name is: {max_pth_file}")
    else:
        print(f"No matching .pth files found in the current folder with '{given_string}'.")
           

# if __name__ == '__main__':
#     # 创建模型、损失函数和优化器
#     # 定义超参数
#     input_dim = 1
#     hidden_dims = [64, 128, 64]  # 可根据需要调整隐藏层的维度和层数
#     output_dim = 1
#     model = FCNModel(input_dim, hidden_dims, output_dim)
#     # model = SimpleModel()
#     criterion = nn.MSELoss()
#     # optimizer = optim.SGD(model.parameters(), lr=0.01)
#     optimizer = optim.Adam(model.parameters(), lr=0.01)

#     # 创建一个虚构的数据加载器
#     # train_loader = torch.utils.data.DataLoader(torch.randn(100, 10), batch_size=10)
#     # test_loader = torch.utils.data.DataLoader(torch.randn(20, 10), batch_size=10)

#     train_dataset = PolynomialDataset(num_points = 10000, method = 'common', ptimes = 2)
#     train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
#     test_dataset = PolynomialDataset(num_points = 15, method = 'random', ptimes = 2)
#     test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

    # 创建Trainer实例
    trainer = Trainer(model, criterion, optimizer, train_loader, test_loader, checkpoint_dir='./cache/')

    # 开始训练，如果存在checkpoint则进行断点续训
    trainer.train(num_epochs=5000, save_interval=1000)
