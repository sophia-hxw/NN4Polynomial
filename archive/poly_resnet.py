import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成二次多项式数据
def generate_data():
    np.random.seed(42)
    x_train = np.random.uniform(-5, 5, 10000)
    # y_train = 2 * x_train**2 - 3 * x_train + 1 + np.random.normal(0, 3, 100)
    y_train = x_train**2
    
    x_test = np.random.uniform(-5, 5, 20)
    # y_test = 2 * x_test**2 - 3 * x_test + 1 + np.random.normal(0, 3, 20)
    y_test = x_test**2
    
    return x_train, y_train, x_test, y_test

# 数据预处理
def preprocess_data(x, y):
    x = torch.from_numpy(x).float().view(-1, 1)
    y = torch.from_numpy(y).float().view(-1, 1)
    # print("x shape =", x.shape)
    # print("y shape =", y.shape)
    return x, y

# 定义简单的Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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

# 训练模型
def train(model, criterion, optimizer, x_train, y_train, batch_size, num_epochs=1000):
    # x_train = x_train.unsqueeze(2)
    for epoch in range(num_epochs):
        model.train()
        # print("epoch = ", epoch)
        for i in range(0, len(x_train), batch_size):
            x_batch = x_train[i:i+batch_size]
            x_batch = x_batch.unsqueeze(2)
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            # print("x_batch shape = ", x_batch.shape)
            outputs = model(x_batch)
            # outputs = model(x_train)
            loss = criterion(outputs, y_batch)
            # loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 测试模型
def test(model, x_test):
    x_test = x_test.unsqueeze(2)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test)
    return y_pred

# 可视化结果
def visualize_results(x, y_true, y_pred):
    plt.scatter(x, y_true, label='True data')
    plt.plot(x, y_pred, color='red', linewidth=3, label='Predicted data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = generate_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    # print("x_train shape = ", x_train.shape)
    # print("y_train shape = ", y_train.shape)
    # print("x_test shape = ", x_test.shape)
    # print("y_test shape = ", y_test.shape)

    model = ResNet(ResidualBlock, [2, 2, 2])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch_size = 100  # 可根据需要调整批处理大小
    train(model, criterion, optimizer, x_train, y_train, batch_size, num_epochs=1000)

    y_pred = test(model, x_test)
    print("x_test = ", x_test)
    print("y_pred = ", y_pred)
    
    # visualize_results(x_test.numpy(), y_test.numpy(), y_pred.numpy())
