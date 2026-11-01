"""实验 6-3 同一任务换成 Transformer 编码器，对比效果和速度。

任务与实验 6-1 相同：看前 30 天的气温，预测第 31 天。
比较 LSTM 和 Transformer 编码器（2 层，4 个头）的参数、误差和训练用时。
运行：python lab6_3_transformer.py（在 labs/ch06 目录下，约 1 分钟）
"""
import sys

import torch
from torch import nn

from common06 import WINDOW, Y_TEST, mae_celsius, predict, train

sys.path.append("..")
from dlbook import set_seed  # noqa: E402


class LSTMModel(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class TransformerModel(nn.Module):
    """每天的气温先变成 32 维向量，加上位置向量，经过 Transformer 编码器，取最后一天的输出做预测。"""

    def __init__(self, dim=32, heads=4, layers=2):
        super().__init__()
        self.inp = nn.Linear(1, dim)
        self.pos = nn.Parameter(torch.zeros(WINDOW, dim))          # 可学习的位置向量
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=64,
                                           dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        h = self.inp(x) + self.pos
        h = self.encoder(h)
        return self.head(h[:, -1]).squeeze(-1)


results = {}
for name, cls in [("LSTM", LSTMModel), ("Transformer", TransformerModel)]:
    set_seed(42)
    model = cls()
    n = sum(p.numel() for p in model.parameters())
    print(f"\n{name}：参数 {n}")
    sec = train(model, epochs=20, lr=0.001 if name == "Transformer" else 0.005, name=name)
    mae = mae_celsius(predict(model), Y_TEST)
    results[name] = (n, mae, sec)

print("\n汇总：")
for name, (n, mae, sec) in results.items():
    print(f"  {name:<12} 参数 {n:>6}  测试集平均误差 {mae:.3f} ℃  训练用时 {sec:.1f} 秒")
