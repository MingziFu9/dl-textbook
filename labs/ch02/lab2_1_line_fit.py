"""实验 2-1 用梯度下降拟合一条直线：广告费与销量

10 家门店的月广告费（万元）和月销量（百件），用 y = w·x + b 去拟合。
从 w = 0、b = 0 出发，按“训练五步”反复调整 w 和 b，看损失怎样一点点变小。
运行：python lab2_1_line_fit.py（在 labs/ch02 目录下）
"""
import sys

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import use_chinese_font  # noqa: E402

# 10 家门店的数据：x 是广告费（万元），y 是销量（百件）
x = torch.tensor([1.0, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0])
y = torch.tensor([7.4, 9.9, 11.0, 12.6, 15.1, 16.0, 17.8, 20.3, 22.4, 25.1])

w = torch.tensor(0.0, requires_grad=True)   # requires_grad=True：告诉 PyTorch 要为它求梯度
b = torch.tensor(0.0, requires_grad=True)
lr = 0.01                                   # 学习率：每次沿下坡方向走多大一步
history = []

for epoch in range(1, 1001):
    pred = w * x + b                        # ① 前向：用当前的 w、b 做预测
    loss = ((pred - y) ** 2).mean()         # ② 算损失：均方误差
    if w.grad is not None:                  # ③ 清梯度：把上一轮留下的梯度清零
        w.grad.zero_()
        b.grad.zero_()
    loss.backward()                         # ④ 反向：PyTorch 自动算出 w、b 的梯度（原理见第 3 章）
    with torch.no_grad():                   # ⑤ 更新：参数 = 参数 − 学习率 × 梯度
        w -= lr * w.grad
        b -= lr * b.grad
    history.append(loss.item())
    if epoch in (1, 2, 3, 10, 50, 100, 500, 1000):
        print(f"第 {epoch:>4} 轮：w = {w.item():.3f}，b = {b.item():.3f}，损失 = {loss.item():.4f}")

print(f"\n拟合结果：销量 ≈ {w.item():.2f} × 广告费 + {b.item():.2f}")
print(f"预测：广告费 5.5 万元时，销量约 {(w * 5.5 + b).item():.1f} 百件")

use_chinese_font()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=120)
a1.plot(range(1, len(history) + 1), history)
a1.set_yscale("log")
a1.set_xlabel("轮次")
a1.set_ylabel("损失（对数刻度）")
a1.set_title("损失曲线")
a1.grid(alpha=0.3)
a2.scatter(x, y, label="门店数据")
xs = torch.linspace(0, 9, 50)
a2.plot(xs, (w * xs + b).detach(), color="C1", label="拟合直线")
a2.set_xlabel("广告费（万元）")
a2.set_ylabel("销量（百件）")
a2.set_title("拟合结果")
a2.legend()
a2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("lab2_1_line_fit.png")
print("图已保存到 lab2_1_line_fit.png")
