"""第 5 章实验 5-2、5-3 共用的数据和训练函数。

数据：CIFAR-10 训练集每类取 2000 张（共 2 万张），测试集用全部 1 万张。
"""
import sys
import time

import torch
from torch import nn

sys.path.append("..")
from dlbook import CIFAR10_CN, load_cifar10, set_seed  # noqa: E402,F401

X_TRAIN, Y_TRAIN = load_cifar10(train=True, n_per_class=2000)
X_TEST, Y_TEST = load_cifar10(train=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, x, y, batch_size=1000):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch_size):
        correct += (model(x[i:i + batch_size]).argmax(1) == y[i:i + batch_size]).sum().item()
    return correct / len(x)


def train(model, epochs=10, lr=0.001, batch_size=64, name="模型"):
    """用 Adam 训练，每轮打印训练损失和测试准确率，返回每轮的测试准确率列表和用时。"""
    set_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    accs, start = [], time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_TRAIN))
        total = 0.0
        for i in range(0, len(X_TRAIN), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X_TRAIN[idx]), Y_TRAIN[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        acc = evaluate(model, X_TEST, Y_TEST)
        accs.append(acc)
        print(f"  [{name}] 第 {epoch:2d} 轮：训练损失 {total / len(X_TRAIN):.4f}，测试准确率 {acc:.4f}")
    return accs, time.time() - start
