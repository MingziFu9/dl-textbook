"""实验 3-1 用自动求导求梯度，并和手算结果对照

① 一个两步的计算 y = (2x + 1)²，在 x = 1 处的导数（3.2.1 节手算为 12）
② 一个神经元 a = w·x + b、损失 L = (a − y)²，求 w 和 b 的梯度（3.2.2 节手算）
③ 第 1 章示范例题的 2-2-1 小网络：让 PyTorch 求出全部 9 个参数的梯度，
   再用“把参数挪一点点，看损失变多少”的笨办法（数值差分）检查一遍
运行：python lab3_1_autograd.py（在 labs/ch03 目录下）
"""
import torch
from torch import nn

# ---------- ① 两步计算 ----------
x = torch.tensor(1.0, requires_grad=True)
u = 2 * x + 1            # 第一步
y = u ** 2               # 第二步
y.backward()             # 从 y 出发往回求导
print("① y = (2x + 1)²，x = 1")
print(f"  前向：u = {u.item()}，y = {y.item()}")
print(f"  自动求导 dy/dx = {x.grad.item()}（手算：2u × 2 = 2 × 3 × 2 = 12）")
print(f"  看看计算图：y 由 {y.grad_fn.name()} 得到，u 由 {u.grad_fn.name()} 得到")

# ---------- ② 一个神经元 ----------
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
x2, target = torch.tensor(2.0), torch.tensor(3.0)
a = w * x2 + b
loss = (a - target) ** 2
loss.backward()
print("\n② 一个神经元：a = w·x + b，L = (a − y)²，w = 0.5，b = 0，x = 2，y = 3")
print(f"  前向：a = {a.item()}，L = {loss.item()}")
print(f"  自动求导：dL/dw = {w.grad.item()}，dL/db = {b.grad.item()}（手算：−8 和 −4）")

# ---------- ③ 2-2-1 小网络 ----------
net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid()).double()   # 用双精度，数值差分更准
with torch.no_grad():
    net[0].weight[:] = torch.tensor([[0.5, -0.2], [-0.3, 0.4]])  # 与第 1 章 1.6 节相同
    net[0].bias[:] = torch.tensor([0.1, -0.1])
    net[2].weight[:] = torch.tensor([[1.0, -0.5]])
    net[2].bias[:] = torch.tensor([0.3])
xin, yt = torch.tensor([[1.0, 2.0]], dtype=torch.float64), torch.tensor([[1.0]], dtype=torch.float64)   # 输入 (1, 2)，正确答案是 1
loss_fn = nn.MSELoss()


def compute_loss():
    return loss_fn(net(xin), yt)


loss = compute_loss()
net.zero_grad()
loss.backward()
print(f"\n③ 2-2-1 小网络：输出 {net(xin).item():.4f}，目标 1，损失（均方误差）{loss.item():.4f}")
print("  参数                       自动求导        数值差分        差别")
eps = 1e-6
names = {"0.weight": "隐藏层权重", "0.bias": "隐藏层偏置", "2.weight": "输出层权重", "2.bias": "输出层偏置"}
max_diff = 0.0
for name, p in net.named_parameters():
    for idx in range(p.numel()):
        with torch.no_grad():
            flat = p.view(-1)
            old = flat[idx].item()
            flat[idx] = old + eps
            up = compute_loss().item()
            flat[idx] = old - eps
            down = compute_loss().item()
            flat[idx] = old
        num = (up - down) / (2 * eps)            # 数值差分：(L(参数+ε) − L(参数−ε)) / 2ε
        auto = p.grad.view(-1)[idx].item()
        max_diff = max(max_diff, abs(num - auto))
        print(f"  {names[name]}[{idx}]{'':<14}{auto:>12.6f}    {num:>12.6f}    {abs(num - auto):.1e}")
print(f"  最大差别 {max_diff:.1e}：两种方法一致，说明自动求导算对了")
