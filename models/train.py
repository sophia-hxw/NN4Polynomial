import torch
import datetime
import torch.nn as nn
import torch.optim as optim
import os
import sys
sys.path.append("../")  # 添加project文件夹到sys.path
from utils.util import find_max_epoch_file

def trainModel(train_loader, model, device = None, num_epochs = 1000, criter = 'MSE', lr=0.01, model_type = None):
    ### train model
    if criter == 'MSE':
        criterion = nn.MSELoss()
    elif criter == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif criter == 'L1':
        criterion = nn.L1Loss()
    else:
        print("Use defaule MSELoss")
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            if model_type == 'cnn':
                labels = labels.unsqueeze(2)
            if device:
                inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()# 把梯度置零，也就是把loss关于weight的导数变成0
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

def testModel(test_loader, model, device = None, criter = 'MSE', model_type = None):
    if criter == 'MSE':
        criterion = nn.MSELoss()
    else:
        print("Use defaule MSELoss")
        criterion = nn.MSELoss()

    model.eval()
    test_losses = []
    pred_y = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            if model_type == 'cnn':
                labels = labels.unsqueeze(2)
            if device:
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            test_losses.append(test_loss.item())
            pred_y.append(outputs.item())
    
    return pred_y, test_losses

def saveModel(model, file_name = './res/Dense_2023-12-21-11-18'):
    # 保存模型
    torch.save(model, file_name + '.pth')

class Trainer:
    def __init__(self, model, train_loader, lr=0.01, criter = 'MSE', checkpoint_dir = './cache/', model_type = None):
        self.model = model
        if criter == 'MSE':
            self.criterion = nn.MSELoss()
        elif criter == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif criter == 'L1':
            self.criterion = nn.L1Loss()
        else:
            print("Use defaule MSELoss")
            self.criterion = nn.MSELoss()
        
        self.optimizer = optim.Adam(model.parameters(), lr)
        self.train_loader = train_loader
        self.checkpoint_dir = checkpoint_dir
        self.model_type = model_type
        """
        os.makedirs(name, mode=0o777, exist_ok=False)：用来创建多层目录（单层请用os.mkdir)
            name：你想创建的目录名
            mode：要为目录设置的权限数字模式，默认的模式为 0o777 (八进制)。
            exist_ok：是否在目录存在时触发异常。如果exist_ok为False（默认值），
                    则在目标目录已存在的情况下触发FileExistsError异常；
                    如果exist_ok为True，则在目标目录已存在的情况下不会触发FileExistsError异常。
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

    def train(self, num_epochs = 5000, save_interval = 1000, device = None):
        # checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        checkpoint_name = find_max_epoch_file(self.checkpoint_dir, search_string = self.model_type)
        print("checkpoint_name: ", checkpoint_name)

        if checkpoint_name:
            # 如果存在checkpoint，加载模型参数
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint.get('model_state_dict', self.model.state_dict()))
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', self.optimizer.state_dict()))
            start_epoch = checkpoint.get('epoch', 0)
            # best_loss = checkpoint.get('loss', float('inf'))
        else:# None
            start_epoch = 0
            # best_loss = float('inf')

        for epoch in range(start_epoch, start_epoch + num_epochs):
            # self.model.train()
            for inputs, targets in self.train_loader:
                if self.model_type == 'CNN':
                    targets = targets.unsqueeze(2)
                if device:
                    inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if (epoch + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

            # 保存checkpoint
            save_checkpoint = os.path.join(self.checkpoint_dir, self.model_type+'_'+str(epoch + 1)+'_'+'checkpoint.pth')
            if (epoch + 1) % save_interval == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, save_checkpoint)