"""实验 2-3 改学习率和批大小，填记录表

为了快一些，只用 1 万张训练图片、训练 3 轮；网络和实验 2-2 相同。
先固定批大小 64，比较 5 种学习率；再固定学习率 0.1，比较 3 种批大小。
运行：python lab2_3_lr_batch.py（在 labs/ch02 目录下，约 1 分钟）
"""
import math
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from dlbook import load_mnist, set_seed  # noqa: E402

x_train, y_train = load_mnist(train=True)
x_train, y_train = x_train[:10000], y_train[:10000]
x_test, y_test = load_mnist(train=False)


def run(lr, batch_size, epochs=3):
    set_seed(42)
    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    start = time.perf_counter()
    losses = []
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
        losses.append(total / len(x_train))
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean().item()
    return losses, acc, time.perf_counter() - start, len(loader) * epochs


def fmt(v):
    return "nan（发散）" if math.isnan(v) else f"{v:.4f}"


print("① 批大小 64，改学习率（每轮的训练损失 → 测试准确率）")
for lr in [0.001, 0.01, 0.1, 1.0, 5.0]:
    losses, acc, sec, steps = run(lr, 64)
    print(f"  学习率 {lr:<5}：损失 {' → '.join(fmt(v) for v in losses)}；测试准确率 {acc:.4f}")

print("\n② 学习率 0.1，改批大小")
for bs in [8, 64, 512]:
    losses, acc, sec, steps = run(0.1, bs)
    print(f"  批大小 {bs:<3}：更新 {steps:>4} 次，用时 {sec:.1f} 秒，最后一轮损失 {fmt(losses[-1])}，测试准确率 {acc:.4f}")
