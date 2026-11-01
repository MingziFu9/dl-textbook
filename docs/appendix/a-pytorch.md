---
title: 附录 A PyTorch 速查
description: 本书实验中用到的 PyTorch 张量操作、网络层、损失函数、优化器、训练五步模板、保存与读取模型，以及常见报错速查
---

# 附录 A PyTorch 速查

本附录汇总了本书实验中用到的 PyTorch 写法，供做实验时随时查阅。它不是完整的 PyTorch 教程，只收本书真正用过的内容；每一项后面注明首次出现的章节，想看详细解释就回到那一节。

## A.1 张量

张量就是“多维数组”，可以看作 NumPy 数组的 PyTorch 版本，能自动求导，也能放到显卡上算（[2.3 节](../chapters/02-training/index.md#23)）。

```python
import torch

a = torch.tensor([1.0, 2.0, 3.0])        # 从列表创建
b = torch.zeros(2, 3)                    # 2 行 3 列的全 0
c = torch.ones(2, 3)                     # 全 1
r = torch.randn(4, 3)                    # 标准正态分布的随机数
torch.arange(0, 10, 2)                   # 0, 2, 4, 6, 8
torch.linspace(0, 1, 5)                  # 0 到 1 等分 5 个数

r.shape                                  # 形状 -> torch.Size([4, 3])
r.dtype                                  # 数据类型 -> torch.float32
r[0], r[:, 1], r[1:3]                    # 取第 0 行、第 1 列、第 1~2 行
r.reshape(3, 4), r.view(12)              # 改形状（元素个数不变）
r.unsqueeze(0).shape                     # 加一个维度 -> [1, 4, 3]
r.flatten(1)                             # 从第 1 维开始摊平

a + 1, a * 2, a ** 2                     # 逐个元素运算
a @ a                                    # 向量点积；矩阵乘法也用 @
r.sum(), r.mean(), r.max()               # 求和、平均、最大值
r.argmax(dim=1)                          # 每行最大值的位置（分类时取预测类别）
torch.cat([b, c], dim=0)                 # 按行拼接 -> 4 行 3 列
torch.stack([a, a])                      # 叠成新的一维 -> 2 行 3 列

a.numpy()                                # 张量 -> NumPy 数组
torch.from_numpy(arr)                    # NumPy 数组 -> 张量
loss.item()                              # 只有一个数的张量 -> Python 数字
```

## A.2 自动求导

给张量加上 `requires_grad=True`，PyTorch 会记下计算图，调用 `backward()` 后自动算出梯度（[3.3 节](../chapters/03-backprop/index.md#33)）。

```python
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x, y = 2.0, 3.0
loss = (w * x + b - y) ** 2              # 前向：算损失
loss.backward()                          # 反向：自动求梯度
print(w.grad, b.grad)                    # tensor(-8.) tensor(-4.)，与 3.2.2 节手算相同

with torch.no_grad():                    # 只算、不记录计算图（更新参数、评估时用）
    w -= 0.05 * w.grad
w.grad.zero_()                           # 梯度会累加，下次反向前要清零
```

## A.3 搭网络

最简单的写法是 `nn.Sequential`，把各层按顺序排好（[2.4 节](../chapters/02-training/index.md#24)）；需要自己安排数据怎样流动时，就继承 `nn.Module`，在 `forward` 里写清楚（[5.5 节](../chapters/05-cnn/index.md#55)）。

```python
from torch import nn

# 写法一：按顺序堆叠
model = nn.Sequential(
    nn.Flatten(),                 # 把 28×28 的图摊平成 784 个数
    nn.Linear(784, 128), nn.ReLU(),
    nn.Linear(128, 10),           # 10 类，输出 10 个分数（不加 Softmax，损失函数里已经包含）
)

# 写法二：继承 nn.Module
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
                                        nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))

model = SmallCNN()
print(model)                                                   # 看结构
print(sum(p.numel() for p in model.parameters()))              # 数参数个数
print(model(torch.randn(1, 3, 32, 32)).shape)                  # 喂一张假图片看输出形状
```

本书用到的层：

| 层 | 作用 | 常用写法 | 章节 |
|---|---|---|---|
| `nn.Linear` | 全连接层 | `nn.Linear(输入个数, 输出个数)` | 第 1 章 |
| `nn.ReLU`、`nn.Sigmoid`、`nn.Tanh`、`nn.LeakyReLU` | 激活函数 | `nn.ReLU()` | 第 1、3 章 |
| `nn.Flatten` | 摊平 | `nn.Flatten()` | 第 2 章 |
| `nn.Dropout` | 训练时随机让一部分神经元休息 | `nn.Dropout(0.5)` | 第 4 章 |
| `nn.BatchNorm1d`、`nn.BatchNorm2d` | 批归一化 | `nn.BatchNorm2d(通道数)` | 第 3、5 章 |
| `nn.Conv2d` | 卷积层 | `nn.Conv2d(输入通道, 卷积核个数, kernel_size=3, stride=1, padding=1)` | 第 5 章 |
| `nn.MaxPool2d` | 最大池化 | `nn.MaxPool2d(2)` | 第 5 章 |
| `nn.AdaptiveAvgPool2d` | 全局平均池化 | `nn.AdaptiveAvgPool2d(1)` | 第 7 章 |
| `nn.RNN`、`nn.LSTM` | 循环层 | `nn.LSTM(输入维度, 隐藏维度, batch_first=True)` | 第 6 章 |
| `nn.MultiheadAttention` | 多头注意力 | `nn.MultiheadAttention(维度, 头数, batch_first=True)` | 第 6 章 |
| `nn.TransformerEncoderLayer`、`nn.TransformerEncoder` | Transformer 编码器 | `nn.TransformerEncoder(层, num_layers=2)` | 第 6 章 |
| `nn.Embedding` | 词嵌入：把编号变成向量 | `nn.Embedding(词表大小, 向量维度, padding_idx=0)` | 第 6、8 章 |

## A.4 损失函数与优化器

| 任务 | 网络最后一层 | 损失函数 | 章节 |
|---|---|---|---|
| 预测一个数（回归） | `nn.Linear(…, 1)` | `nn.MSELoss()` | 第 2 章 |
| 多选一（多分类） | `nn.Linear(…, 类别数)`，不加 Softmax | `nn.CrossEntropyLoss()`，标签是类别编号 | 第 2 章 |
| 是或否（二分类） | `nn.Linear(…, 1)`，不加 Sigmoid | `nn.BCEWithLogitsLoss()`，标签是 0.0 或 1.0 | 第 9 章 |

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)                 # 小批量梯度下降
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 带动量
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)              # Adam，本书默认
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)   # 微调预训练模型
torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)       # 加权重衰减
```

## A.5 数据加载

```python
from torch.utils.data import DataLoader, TensorDataset

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
for xb, yb in train_loader:        # 每次取出一批
    print(xb.shape, yb.shape)       # 例如 torch.Size([64, 1, 28, 28]) torch.Size([64])
    break

idx = torch.randperm(len(x))        # 打乱顺序，用来自己划分训练集和验证集
x_val, x_train = x[idx[:1000]], x[idx[1000:]]
```

图像数据增强（[4.4.1 节](../chapters/04-generalization/index.md#441)、[7.2 节](../chapters/07-image-transfer/index.md#72)）：

```python
from torchvision import transforms
augment = transforms.Compose([
    transforms.RandomResizedCrop(112, scale=(0.6, 1.0)),   # 随机裁剪一块再缩放
    transforms.RandomHorizontalFlip(),                     # 随机左右翻转
])
xb = augment(xb)                                           # 只在训练时用，验证和测试不用
```

## A.6 训练五步模板

本书所有训练程序都是这个结构（[2.4 节](../chapters/02-training/index.md#24)）：

```python
for epoch in range(1, 11):              # 训练 10 轮
    model.train()                       # 训练模式（Dropout、批归一化生效）
    for xb, yb in train_loader:
        out = model(xb)                 # ① 前向
        loss = loss_fn(out, yb)         # ② 算损失
        optimizer.zero_grad()           # ③ 清梯度
        loss.backward()                 # ④ 反向
        optimizer.step()                # ⑤ 更新

    model.eval()                        # 评估模式
    with torch.no_grad():               # 评估时不需要梯度
        pred = model(x_val).argmax(1)
        acc = (pred == y_val).float().mean().item()
    print(f"第 {epoch} 轮，验证准确率 {acc:.4f}")
```

## A.7 保存、读取与迁移学习

```python
torch.save(model.state_dict(), "my_model.pt")      # 只保存参数（推荐）

model = SmallCNN()                                 # 先搭一个结构相同的网络
model.load_state_dict(torch.load("my_model.pt", weights_only=True))
model.eval()                                       # 读回后先切到评估模式再预测
```

`weights_only=True` 表示只读取参数，不执行文件里的任何代码。从网上下载的模型文件尤其要这样读取。

迁移学习（[7.3 节](../chapters/07-image-transfer/index.md#73)）：

```python
from torchvision.models import resnet18
net = resnet18()                                    # 搭 ResNet18 结构
net.load_state_dict(torch.load("resnet18_a1_0-d63eafa0.pth", weights_only=True))   # 读预训练参数
for p in net.parameters():
    p.requires_grad = False                         # 冻结全部层
net.fc = nn.Linear(net.fc.in_features, 10)          # 换成自己的 10 类输出层（新层默认可训练）
optimizer = torch.optim.Adam(net.fc.parameters(), lr=0.001)   # 只训练最后一层
```

## A.8 常见报错速查

| 报错信息（节选） | 常见原因 | 解决办法 |
|---|---|---|
| `mat1 and mat2 shapes cannot be multiplied (64x2048 and 1024x128)` | `nn.Linear` 的输入个数和摊平后的实际长度对不上 | 逐层打印形状，把 `Flatten` 后的长度填到 `nn.Linear` 第一个参数（[5.3.2 节](../chapters/05-cnn/index.md#532)） |
| `Expected input batch_size (…) to match target batch_size (…)` | 输出和标签的个数不同，多半是摊平写错了 | 检查 `model(xb).shape` 和 `yb.shape` 的第一维 |
| `expected scalar type Long but found Float` | `CrossEntropyLoss` 的标签必须是整数类别编号 | `yb = yb.long()` |
| `Found dtype Long but expected Float` | `MSELoss`、`BCEWithLogitsLoss` 的标签必须是小数 | `yb = yb.float()` |
| `Given groups=1, weight of size [16, 3, 3, 3], expected input … to have 3 channels` | 输入图片的通道数和 `Conv2d` 第一个参数不同，或者少了批这一维 | 灰度图用 1 通道；单张图片用 `x.unsqueeze(0)` 加上批维度 |
| `element 0 of tensors does not require grad` | 参数被冻结或在 `torch.no_grad()` 里算了损失 | 检查 `requires_grad` 和 `no_grad` 的范围 |
| `Can't call numpy() on Tensor that requires grad` | 要转成 NumPy 的张量还连着计算图 | `x.detach().numpy()` |
| 损失变成 `nan` | 学习率太大、输入没有标准化 | 学习率缩小 10 倍，检查数据（[2.2.2 节](../chapters/02-training/index.md#222)） |
| `CUDA out of memory`（用显卡时） | 批太大、图片太大 | 减小 `batch_size` 或图片尺寸 |
