import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 编码器
def encode_input(input_str):
    tokens = input_str.split()
    encoding = [5]  # "<eos>" 标记
    for token in tokens:
        if token == "AND":
            encoding.append(0)
        elif token == "OR":
            encoding.append(1)
        elif token == "NOT":
            encoding.append(2)
        elif token == "True":
            encoding.append(3)
        elif token == "False":
            encoding.append(4)
    return torch.tensor(encoding, dtype=torch.long).view(1, -1)
# 布尔计算类
class BoolDataset(Dataset):
    def __init__(self, expressions, targets):
        self.expressions = expressions
        self.targets = targets

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, index):
        return self.expressions[index], self.targets[index]


# Transformer 模型
class BoolTransformer(nn.Module):
    def __init__(self, d_model=16, nhead=4):
        super().__init__()
        self.transformer = nn.Transformer(d_model, nhead=nhead)
        self.embedding = nn.Embedding(6, d_model)
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x[:-1], x[1:])
        x = self.fc_out(x)
        return torch.sigmoid(x)

# OR 和 NOT 训练示例
train_data = ["OR True False", "OR False True", "OR True True", "OR False False",
              "NOT True", "NOT False"]
train_targets = [True, True, True, False, False, True]

# 表达式和目标编码
train_expressions = [encode_input(expr) for expr in train_data]
train_targets = [torch.tensor(target, dtype=torch.float32) for target in train_targets]

# 创建数据集和数据加载器
train_dataset = BoolDataset(train_expressions, train_targets)
print(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=1)

model = BoolTransformer()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# 开始训练
num_epochs = 1000
for epoch in range(num_epochs):
    epoch_loss = 0
    for data, target in train_dataloader:
        print(data, target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    if epoch % 100 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, num_epochs, epoch_loss / len(train_dataloader)))


test_data = "OR True False"
enc_input = encode_input(test_data)
prediction = model(enc_input)
print(f"Prediction for {test_data}: {prediction.item() > 0.5}")
