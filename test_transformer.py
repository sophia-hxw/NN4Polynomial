import torch
from torch import nn
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import matplotlib.pyplot as plt

# 生成多项式数据集
def generate_polynomial_data(poly_degree, num_samples):
    x = torch.linspace(-1, 1, num_samples)
    coefficients = torch.randn(poly_degree + 1)
    y = sum(coefficients[i] * x**i for i in range(poly_degree + 1))
    return x, y

# 准备数据
poly_degree = 2
num_samples = 100
x, y = generate_polynomial_data(poly_degree, num_samples)

# 使用GPT-2的tokenizer对输入进行编码
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
encoded_input = tokenizer(x.numpy().tolist(), return_tensors='pt', padding=True, truncation=True)

# 定义Transformer模型
class PolynomialTransformer(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolynomialTransformer, self).__init__()
        self.transformer = GPT2Model.from_pretrained('gpt2')
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input_ids, attention_mask):
        transformer_output = self.transformer(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = transformer_output.mean(dim=1)
        output = self.fc(pooled_output)
        return output

# 训练模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolynomialTransformer(input_size=768, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

input_ids = encoded_input['input_ids'].to(device)
attention_mask = encoded_input['attention_mask'].to(device)
y = y.view(-1, 1).float().to(device)

num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

# 评估模型
model.eval()
with torch.no_grad():
    test_x = torch.linspace(-1, 1, 50).to(device)
    encoded_test_input = tokenizer(test_x.cpu().numpy().tolist(), return_tensors='pt', padding=True, truncation=True)
    test_input_ids = encoded_test_input['input_ids'].to(device)
    test_attention_mask = encoded_test_input['attention_mask'].to(device)

    predictions = model(test_input_ids, test_attention_mask)

# 绘制结果
plt.scatter(x.numpy(), y.cpu().numpy(), label='Actual Data')
plt.plot(test_x.cpu().numpy(), predictions.cpu().numpy(), label='Predictions', color='red')
plt.legend()
plt.show()

plt.savefig("test_transformer.png")
