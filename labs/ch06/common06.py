"""第 6 章实验 6-1、6-3 共用的气温数据和训练函数。

数据：墨尔本 1981—1990 年每日最低气温，共 3650 天。
用前 8 年（1981—1988，2920 天）训练，后 2 年（1989—1990，730 天）测试。
任务：看前 30 天的气温，预测第 31 天的气温。
"""
import sys
import time

import torch
from torch import nn

sys.path.append("..")
from dlbook import load_temperature, set_seed  # noqa: E402

WINDOW = 30
DATES, TEMPS = load_temperature()
N_TRAIN = 2920
MEAN, STD = TEMPS[:N_TRAIN].mean(), TEMPS[:N_TRAIN].std()     # 只用训练部分算平均值和标准差
Z = torch.tensor((TEMPS - MEAN) / STD)                         # 标准化：变成平均 0、标准差 1 左右


def make_windows(start, end):
    """目标日期在 [start, end) 之间的所有样本：输入是前 30 天，形状 N×30×1；输出是当天，形状 N。"""
    xs = torch.stack([Z[t - WINDOW:t] for t in range(max(start, WINDOW), end)])
    ys = torch.stack([Z[t] for t in range(max(start, WINDOW), end)])
    return xs.unsqueeze(-1), ys


X_TRAIN, Y_TRAIN = make_windows(0, N_TRAIN)
X_TEST, Y_TEST = make_windows(N_TRAIN, len(Z))


def mae_celsius(pred, y):
    """平均绝对误差，换算回摄氏度。"""
    return float((pred - y).abs().mean() * STD)


def train(model, epochs=20, lr=0.005, batch_size=64, name="模型", verbose=True):
    set_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_TRAIN))
        for i in range(0, len(X_TRAIN), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X_TRAIN[idx]), Y_TRAIN[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  [{name}] 第 {epoch:2d} 轮：测试集平均误差 {mae_celsius(predict(model), Y_TEST):.3f} ℃")
    return time.time() - start


@torch.no_grad()
def predict(model, x=None):
    model.eval()
    return model(X_TEST if x is None else x)
