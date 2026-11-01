"""编者用来生成 mnist_mlp.pt 的脚本（学生不必运行）。

训练一个“784 → 128 → 64 → 10”的多层感知机识别 MNIST 手写数字，权重保存为 mnist_mlp.pt，
供实验 1-2 直接加载使用。训练方法在第 2 章学习。
"""
import sys

import torch
from torch import nn

sys.path.append("..")
from dlbook import load_mnist, set_seed  # noqa: E402

set_seed(42)
x, y = load_mnist(train=True)
model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 10))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(5):
    perm = torch.randperm(len(x))
    for i in range(0, len(x), 128):
        idx = perm[i:i + 128]
        loss = loss_fn(model(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"第 {epoch + 1} 轮完成，最后一批损失 {loss.item():.4f}")
torch.save(model.state_dict(), "mnist_mlp.pt")
print("已保存 mnist_mlp.pt")
