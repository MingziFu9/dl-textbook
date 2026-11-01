"""实验 6-1 用 LSTM 预测下一天的气温。

看前 30 天的最低气温，预测第 31 天。比较三种做法：
① 最简单的猜法“明天和今天一样”；② 简单 RNN；③ LSTM。
运行：python lab6_1_lstm_temperature.py（在 labs/ch06 目录下，约 1 分钟）
"""
import sys

import matplotlib
from torch import nn

from common06 import DATES, MEAN, N_TRAIN, STD, TEMPS, WINDOW, X_TEST, X_TRAIN, Y_TEST, mae_celsius, predict, train

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import set_seed, use_chinese_font  # noqa: E402

print(f"共 {len(TEMPS)} 天，{DATES[0]} 到 {DATES[-1]}；训练部分平均 {MEAN:.2f} ℃，标准差 {STD:.2f} ℃")
print(f"训练样本 {tuple(X_TRAIN.shape)}，测试样本 {tuple(X_TEST.shape)}（样本数 × 天数 × 每天 1 个数）")

# ---------- ① 最简单的猜法：明天和今天一样 ----------
naive = X_TEST[:, -1, 0]                     # 每个样本输入的最后一天
print(f"\n① 猜“明天和今天一样”：平均误差 {mae_celsius(naive, Y_TEST):.3f} ℃")


class SeqModel(nn.Module):
    """循环层读完 30 天，取最后一步的隐藏状态，接一个全连接层输出一个数。"""

    def __init__(self, cell="lstm", hidden=32):
        super().__init__()
        rnn_class = nn.LSTM if cell == "lstm" else nn.RNN
        self.rnn = rnn_class(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                    # x 形状：批量 × 30 × 1
        out, _ = self.rnn(x)                 # out 形状：批量 × 30 × hidden，每一天都有一个隐藏状态
        return self.head(out[:, -1]).squeeze(-1)   # 只用最后一天的隐藏状态做预测


results = {}
for cell, name in [("rnn", "② 简单 RNN"), ("lstm", "③ LSTM")]:
    set_seed(42)
    model = SeqModel(cell)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{name}：参数 {n_params}")
    sec = train(model, epochs=20, name=name)
    pred = predict(model)
    results[name] = (mae_celsius(pred, Y_TEST), sec, pred)
    print(f"{name}：测试集平均误差 {results[name][0]:.3f} ℃，用时 {sec:.0f} 秒")

# ---------- 画图：测试期前 120 天 ----------
use_chinese_font()
lstm_pred = results["③ LSTM"][2] * STD + MEAN
true = Y_TEST * STD + MEAN
fig, ax = plt.subplots(figsize=(8, 3.4), dpi=120)
ax.plot(true[:120].numpy(), label="真实气温", color="#263238")
ax.plot(lstm_pred[:120].numpy(), label="LSTM 预测", color="#ef6c00")
ax.set_xlabel(f"天（从 {DATES[N_TRAIN]} 起）")
ax.set_ylabel("最低气温 ℃")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("lab6_1_forecast.png")
print("\n预测曲线已保存到 lab6_1_forecast.png")

print("\n测试期前 5 天：")
for i in range(5):
    print(f"  {DATES[N_TRAIN + i]}：真实 {true[i]:.1f} ℃，LSTM 预测 {lstm_pred[i]:.1f} ℃")
_ = WINDOW
