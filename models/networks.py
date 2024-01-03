import torch.nn as nn

# 定义cnn网络
class CNN(nn.Module):
    def __init__(self, activate = 'LeakyReLU'):
        super(CNN, self).__init__()
        if activate == 'LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif activate == 'Tanh':
            self.relu = nn.Tanh()
        elif activate == 'Sigmoid':
            self.relu = nn.Sigmoid()
        elif activate == 'Mish':
            self.relu = nn.Mish()
        else:
            print("Default activation is ReLU")
            self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(1, 20, kernel_size=1)
        self.conv2 = nn.Conv1d(20, 40, kernel_size=1)
        self.conv3 = nn.Conv1d(40, 80, kernel_size=1)
        self.conv4 = nn.Conv1d(80, 160, kernel_size=1)
        self.conv5 = nn.Conv1d(160, 1, kernel_size=1)

    def forward(self, x):
        x = x.unsqueeze(2)
        # print("original x = ", x.shape)
        x = self.conv1(x)
        # print("conv1 x = ", x.shape)
        x = self.relu(x)
        # print("relu1 x = ", x.shape)
        x = self.conv2(x)
        # print("conv2 x = ", x.shape)
        x = self.relu(x)
        # print("relu2 x = ", x.shape)
        x = self.conv3(x)
        # print("conv3 x = ", x.shape)
        x = self.relu(x)
        # print("relu3 x = ", x.shape)
        x = self.conv4(x)
        # print("conv4 x = ", x.shape)
        x = self.relu(x)
        # print("relu4 x = ", x.shape)
        x = self.conv5(x)
        # print("conv5 x = ", x.shape)
        return x

# 定义 Dense 网络
class DenseNet(nn.Module):
    def __init__(self, activate = 'LeakyReLU'):
        super(DenseNet, self).__init__()
        if activate == 'LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif activate == 'Tanh':
            self.relu = nn.Tanh()
        elif activate == 'Sigmoid':
            self.relu = nn.Sigmoid()
        elif activate == 'Mish':
            self.relu = nn.Mish()
        else:
            print("Default activation is ReLU")
            self.relu = nn.ReLU()
            
        self.fc1 = nn.Linear(1, 32)  # 输入维度为1，输出维度为64
        self.fc2 = nn.Linear(32, 32)  # 输入维度为64，输出维度为64
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 64)
        self.fc5 = nn.Linear(64, 1)  # 输入维度为64，输出维度为1

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        x = self.fc5(x)
        return x

# 定义简单的Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 使用1x1卷积进行通道匹配
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.downsample(identity)
        out = self.relu(out)
        
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.conv1.shape = [16,1] # pytorch.shape=[NCWH]
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0], stride=1)
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("x shape = ", x.shape)
        x = x.unsqueeze(2)
        out = self.conv1(x)
        # print("out1 shape = ", out.shape)
        out = self.bn1(out)
        # print("out2 shape = ", out.shape)
        out = self.relu(out)
        # print("out3 shape = ", out.shape)
        out = self.layer1(out)
        # print("out4 shape = ", out.shape)
        out = self.layer2(out)
        # print("out5 shape = ", out.shape)
        out = self.layer3(out)
        # print("out6 shape = ", out.shape)
        out = self.avg_pool(out)
        # print("out7 shape = ", out.shape)
        out = out.view(out.size(0), -1)
        # print("out8 shape = ", out.shape)
        out = self.fc(out)
        return out

# 定义FCN模型
class FCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FCNModel, self).__init__()
        if activate == 'LeakyReLU':
            self.relu = nn.LeakyReLU()
        elif activate == 'Tanh':
            self.relu = nn.Tanh()
        elif activate == 'Sigmoid':
            self.relu = nn.Sigmoid()
        elif activate == 'Mish':
            self.relu = nn.Mish()
        else:
            print("Default activation is ReLU")
            self.relu = nn.ReLU()
        
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.fcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.fcn(x)