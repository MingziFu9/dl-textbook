"""第 4 章三个实验共用的数据和训练函数。

数据：从 MNIST 训练集里取 1000 张作训练集、另取 1000 张作验证集，测试集用官方的 1 万张。
故意只用 1000 张训练图片，让过拟合容易出现、便于观察。
"""
import sys

import torch
from torch import nn

sys.path.append("..")
from dlbook import load_mnist, set_seed  # noqa: E402

set_seed(42)
_x, _y = load_mnist(train=True)
_perm = torch.randperm(len(_x))
X_TRAIN, Y_TRAIN = _x[_perm[:1000]], _y[_perm[:1000]]
X_VAL, Y_VAL = _x[_perm[1000:2000]], _y[_perm[1000:2000]]
X_TEST, Y_TEST = load_mnist(train=False)


def make_mlp(hidden=512, dropout=0.0):
    """784 → hidden → hidden → 10 的多层感知机，dropout 大于 0 时在每个隐藏层后加 Dropout。"""
    layers = [nn.Flatten()]
    n_in = 784
    for _ in range(2):
        layers += [nn.Linear(n_in, hidden), nn.ReLU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        n_in = hidden
    layers.append(nn.Linear(n_in, 10))
    return nn.Sequential(*layers)


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        out = model(x)
        return nn.functional.cross_entropy(out, y).item(), (out.argmax(1) == y).float().mean().item()


def fit(model, epochs=60, lr=1e-3, weight_decay=0.0, patience=None, augment=None, batch_size=64, verbose=False):
    """训练并返回每轮的记录。patience 不为空时启用早停：验证损失连续 patience 轮不创新低就停，并恢复最好的参数。
    augment 是一个函数，训练时对每批图片做随机变换（数据增强）。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_TRAIN))
        for i in range(0, len(X_TRAIN), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_TRAIN[idx], Y_TRAIN[idx]
            if augment is not None:
                xb = augment(xb)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        tl, ta = evaluate(model, X_TRAIN, Y_TRAIN)
        vl, va = evaluate(model, X_VAL, Y_VAL)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        if verbose and (epoch in (1, 5, 10, 20, 40, 60) or epoch == epochs):
            print(f"  第 {epoch:>2} 轮：训练损失 {tl:.4f}，验证损失 {vl:.4f}，训练准确率 {ta:.3f}，验证准确率 {va:.3f}")
        if vl < best:
            best, best_epoch, wait = vl, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience is not None and wait >= patience:
                if verbose:
                    print(f"  验证损失已连续 {patience} 轮没有创新低，在第 {epoch} 轮早停")
                break
    if patience is not None:
        model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    return hist
