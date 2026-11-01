"""实验 2-2 训练五步：多层感知机识别手写数字

从零开始训练第 1 章用过的网络（784 → 128 → 64 → 10），完整走一遍
“准备数据 → 搭网络 → 选损失和优化器 → 训练五步循环 → 在测试集上评估 → 保存模型”。
运行：python lab2_2_mnist_mlp.py（在 labs/ch02 目录下）
"""
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from dlbook import load_mnist, plot_curves, set_seed  # noqa: E402

set_seed(42)

# ---------- 1. 准备数据 ----------
x_train, y_train = load_mnist(train=True)
x_test, y_test = load_mnist(train=False)
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)   # 每批 64 张，每轮打乱
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=1000)
print(f"训练集 {len(x_train)} 张，测试集 {len(x_test)} 张；每轮 {len(train_loader)} 批，每批 64 张")
print(f"一批数据的形状：{next(iter(train_loader))[0].shape}")

# ---------- 2. 搭网络 ----------
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))

# ---------- 3. 选损失函数和优化器 ----------
loss_fn = nn.CrossEntropyLoss()                         # 多分类用交叉熵
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 小批量梯度下降，学习率 0.1


def evaluate(loader):
    """在一个数据集上算平均损失和准确率（只预测，不训练）。"""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n


# ---------- 4. 训练：训练五步，重复很多批、很多轮 ----------
history = {"训练损失": [], "测试损失": []}
start = time.perf_counter()
for epoch in range(1, 6):
    model.train()
    running = 0.0
    for xb, yb in train_loader:
        out = model(xb)              # ① 前向
        loss = loss_fn(out, yb)      # ② 算损失
        optimizer.zero_grad()        # ③ 清梯度
        loss.backward()              # ④ 反向
        optimizer.step()             # ⑤ 更新
        running += loss.item() * len(yb)
    train_loss = running / len(x_train)
    test_loss, test_acc = evaluate(test_loader)
    history["训练损失"].append(train_loss)
    history["测试损失"].append(test_loss)
    print(f"第 {epoch} 轮：训练损失 {train_loss:.4f}，测试损失 {test_loss:.4f}，"
          f"测试准确率 {test_acc:.4f}，累计用时 {time.perf_counter() - start:.1f} 秒")

# ---------- 5. 评估与保存 ----------
plot_curves(history, "lab2_2_curves.png", title="多层感知机识别手写数字")
torch.save(model.state_dict(), "my_mnist_mlp.pt")
print("模型参数已保存到 my_mnist_mlp.pt（可以替换实验 1-2 的 mnist_mlp.pt 试试）")
