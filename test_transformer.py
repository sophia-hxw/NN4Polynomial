import torch
from torch import nn
from transformers import GPT2ForSequenceClassification, GPT2Config, GPT2Tokenizer
import matplotlib.pyplot as plt

# 生成多项式数据集
def generate_polynomial_data(poly_degree, num_samples):
    x = torch.linspace(-1, 1, num_samples)
    # coefficients = torch.randn(poly_degree + 1)
    # y = sum(coefficients[i] * x**i for i in range(poly_degree + 1))
    y = [x**i for i in range(poly_degree + 1)]
    return x, y

def see_source_data():
    # 准备数据
    poly_degree = 2
    num_samples = 1000
    x, y = generate_polynomial_data(poly_degree, num_samples)
    # print("x = ", x)
    # print("y = ", y)
    return x, y

# 图示生成的数据，其中y是x^0到x^poly_degree次方
def data_figure(x0, y):
    y0, y1, y2 = y
    # 绘制图形并添加图例
    # supported values are '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    plt.plot(x0, y0, color = 'red', marker='.', linestyle='None', label='x^0', linewidth=0.5)
    plt.plot(x, y1, color = 'blue', marker='.', linestyle='None', label='x^1')
    plt.plot(x, y2, color = 'green', marker='.', linestyle='None', label='x^2')

    # 添加标题和标签
    plt.title("data show")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 添加图例
    plt.legend()

    # 保存图像（可以根据需要更改文件名和格式）
    # plt.savefig("polynomial approximation of exp(x)-" + str(pt_num) + '.png')

    # 显示图形（可选）
    plt.show()

# 定义Transformer模型
class PolynomialTransformer(nn.Module):
    def __init__(self, config, input_size, output_size):
        super(PolynomialTransformer, self).__init__()
        self.transformer = GPT2ForSequenceClassification(config)

    def forward(self, inputs, labels=None):
        transformer_output = self.transformer(inputs, labels=labels).last_hidden_state
        return transformer_output


def test2():
    # 使用GPT-2的tokenizer对输入进行编码
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # 编码输入和输出
    input_text = "Polynomial fit: " + ', '.join([str(val) for val in x.numpy().tolist()])
    encoded_input = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=1024)

    # 调整目标张量的大小以匹配模型的输出大小
    outputs = torch.tensor(y.tolist()).view(-1, 1).float()



def test2():
    # 定义配置
    config = GPT2Config.from_pretrained('gpt2', num_labels=1)

    # 初始化模型
    model = PolynomialTransformer(config, input_size=config.n_embd, output_size=1)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        predictions = model(encoded_input)
        # For simplicity, let's use the mean pooling over the sequence dimension
        pooled_output = torch.mean(predictions, dim=1)
        pooled_output = pooled_output.view(-1, config.n_embd)
        
        # Assuming outputs is a tensor with shape (batch_size, 1)
        loss = criterion(pooled_output[:, -1], outputs)
        
        loss.backward()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # 评估模型
    model.eval()
    with torch.no_grad():
        test_x = torch.linspace(-1, 1, 50).view(1, -1)
        input_text_test = "Polynomial fit: " + ', '.join([str(val) for val in test_x.numpy().tolist()])
        encoded_test_input = tokenizer.encode(input_text_test, return_tensors="pt", truncation=True, max_length=1024)
        predictions = model(encoded_test_input)

    # 绘制结果
    plt.scatter(x.numpy(), y.numpy(), label='Actual Data')
    plt.plot(test_x.numpy(), predictions[0, -1, :].numpy(), label='Predictions', color='red')  # Assuming predictions[0, -1, :] is the last hidden state
    plt.legend()
    plt.show()

if __name__ == "__main__":
    x, y = see_source_data()
    data_figure(x, y)


