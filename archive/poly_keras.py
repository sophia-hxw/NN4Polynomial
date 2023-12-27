import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# 创建一个正弦函数的数据集
X = np.linspace(-np.pi, np.pi, 2000)
Y = np.sin(X)

# 创建一个具有两个隐藏层的神经网络
model = Sequential()
model.add(Dense(10, input_dim=1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='linear'))

# 编译模型
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(X, Y, epochs=1000, verbose=0)

# 在测试集上进行预测
X_test = np.linspace(-np.pi, np.pi, 200)
Y_test = model.predict(X_test)

# 绘制结果
plt.plot(X, Y)
plt.plot(X_test, Y_test)
plt.show()