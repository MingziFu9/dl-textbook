"""实验 3-2 深层网络里看梯度消失，再比一比三种优化器

① 搭四个 10 层的全连接网络：Sigmoid、ReLU、ReLU + 何恺明初始化、ReLU + 批归一化，喂进同一批手写数字，
   求一次梯度，看每一层的梯度有多大（越靠近输入越小，就是梯度消失）
② 四个网络各训练 2 轮，看准确率
③ 同一个 3 层网络，分别用 SGD、带动量的 SGD、Adam 训练 2 轮，比较损失和准确率
运行：python lab3_2_vanishing_optim.py（在 labs/ch03 目录下，约 1 分钟）
"""
import sys

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("..")
from dlbook import load_mnist, set_seed  # noqa: E402

x_train, y_train = load_mnist(train=True)
x_train, y_train = x_train[:20000], y_train[:20000]       # 用 2 万张，快一些
x_test, y_test = load_mnist(train=False)


def deep_net(act, depth=10, width=64, he_init=False, batch_norm=False):
    """depth 个全连接层，每层 width 个神经元，层间用 act 激活。
    he_init=True：按“何恺明初始化”给 ReLU 网络设初始权重；batch_norm=True：每层后加批归一化。"""
    layers, n_in = [nn.Flatten()], 784
    for _ in range(depth - 1):
        layers.append(nn.Linear(n_in, width))
        if batch_norm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(act())
        n_in = width
    layers.append(nn.Linear(n_in, 10))
    model = nn.Sequential(*layers)
    if he_init:
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
    return model


def train(model, opt, epochs=2):
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(yb)
    with torch.no_grad():
        model.eval()
        acc = (model(x_test).argmax(1) == y_test).float().mean().item()
    return total / len(x_train), acc


print("① 10 层网络，同一批 64 张图片，每一层权重梯度的平均大小")
nets = {}
configs = [("Sigmoid", nn.Sigmoid, {}), ("ReLU", nn.ReLU, {}),
           ("ReLU + 何恺明初始化", nn.ReLU, {"he_init": True}), ("ReLU + 批归一化", nn.ReLU, {"batch_norm": True})]
for name, act, kw in configs:
    set_seed(42)
    model = deep_net(act, **kw)
    nets[name] = model
    loss = nn.CrossEntropyLoss()(model(x_train[:64]), y_train[:64])
    loss.backward()
    grads = [m.weight.grad.abs().mean().item() for m in model if isinstance(m, nn.Linear)]
    print(f"  {name}：第 1 层 {grads[0]:.2e}，第 5 层 {grads[4]:.2e}，第 10 层 {grads[9]:.2e}，"
          f"第 1 层只有第 10 层的 {grads[0] / grads[9]:.1e} 倍")

print("\n② 四个 10 层网络各训练 2 轮（SGD，学习率 0.1）")
for name, model in nets.items():
    set_seed(42)
    model.zero_grad()
    loss, acc = train(model, torch.optim.SGD(model.parameters(), lr=0.1))
    print(f"  {name}：最后一轮训练损失 {loss:.4f}，测试准确率 {acc:.4f}")

print("\n③ 3 层网络（784 → 128 → 128 → 10，ReLU），三种优化器各训练 2 轮")
for name in ["SGD", "SGD + 动量", "Adam"]:
    set_seed(42)
    model = deep_net(nn.ReLU, depth=3, width=128)
    if name == "SGD":
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
    elif name == "SGD + 动量":
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss, acc = train(model, opt)
    print(f"  {name:<9}：最后一轮训练损失 {loss:.4f}，测试准确率 {acc:.4f}")
