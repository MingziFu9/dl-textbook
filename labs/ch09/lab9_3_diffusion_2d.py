"""实验 9-3 扩散模型体验：在二维平面上“一点点加噪声，再一点点去掉”。

图片的扩散模型要用 GPU 训练很久。这里用同样的方法处理最简单的数据：平面上排成“笑脸”形状的 2000 个点。
1. 正向：给点一步步加噪声，200 步后变成一团乱点；
2. 训练：一个小网络学习“看一眼加了噪声的点和步数，猜出加了什么噪声”；
3. 反向：从一团随机点出发，用网络一步步去噪，看能不能变回笑脸。
运行：python lab9_3_diffusion_2d.py（在 labs/ch09 目录下，约 1 分钟）
"""
import math
import sys
import time

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import set_seed, use_chinese_font  # noqa: E402

use_chinese_font()
set_seed(42)


def smiley(n):
    """生成笑脸形状的点：一个大圆（脸）、两个小圆（眼睛）、一段弧（嘴）。"""
    parts = []
    t = torch.rand(n // 2) * 2 * math.pi
    parts.append(torch.stack([torch.cos(t), torch.sin(t)], 1))                       # 脸
    for cx in (-0.35, 0.35):
        t = torch.rand(n // 8) * 2 * math.pi
        parts.append(torch.stack([cx + 0.1 * torch.cos(t), 0.35 + 0.1 * torch.sin(t)], 1))   # 眼睛
    t = math.pi * (1.15 + 0.7 * torch.rand(n - n // 2 - 2 * (n // 8)))
    parts.append(torch.stack([0.5 * torch.cos(t), 0.5 * torch.sin(t)], 1))         # 嘴
    return torch.cat(parts) + 0.02 * torch.randn(n, 2)


data = smiley(2000)

# ---------- 1. 正向加噪（共 200 步）：每一步保留一点原来的点，再混入一点噪声 ----------
T = 200
betas = torch.linspace(1e-4, 0.06, T)              # 每一步加多少噪声
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, 0)               # 到第 t 步，原来的点还剩多少“分量”


def add_noise(x0, t, noise):
    """直接算出第 t 步加噪后的点：x_t = √ᾱ_t · x0 + √(1−ᾱ_t) · 噪声。"""
    a = alpha_bar[t].unsqueeze(1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


print("第 t 步时原来的点还剩多少：", ", ".join(f"t={t}: {alpha_bar[t - 1].sqrt():.3f}" for t in (1, 20, 50, 100, 200)))


# ---------- 2. 训练去噪网络：输入加噪的点和步数，输出猜测的噪声 ----------
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_emb = nn.Embedding(T, 32)           # 每个步数一个向量，告诉网络“现在噪声有多大”
        self.net = nn.Sequential(nn.Linear(2 + 32, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, x, t):
        return self.net(torch.cat([x, self.t_emb(t)], 1))


model = Denoiser()
opt = torch.optim.Adam(model.parameters(), lr=0.001)
start = time.time()
for step in range(1, 15001):
    idx = torch.randint(0, len(data), (512,))
    x0 = data[idx]
    t = torch.randint(0, T, (512,))                # 随机挑一个步数
    noise = torch.randn_like(x0)
    loss = ((model(add_noise(x0, t, noise), t) - noise) ** 2).mean()   # 猜得越准，损失越小
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 3000 == 0:
        print(f"  训练第 {step} 步：损失 {loss.item():.4f}，已用 {time.time() - start:.0f} 秒")


# ---------- 3. 反向去噪：从纯随机点出发，一步步减去网络猜出的噪声 ----------
@torch.no_grad()
def sample(n=2000):
    x = torch.randn(n, 2)
    history = {T: x.clone()}
    for t in reversed(range(T)):
        tt = torch.full((n,), t)
        eps = model(x, tt)
        x = (x - betas[t] / (1 - alpha_bar[t]).sqrt() * eps) / alphas[t].sqrt()
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
        history[t] = x.clone()
    return history


set_seed(0)
hist = sample()
steps = [0, 50, 100, 150, 200]
fig, axes = plt.subplots(2, 5, figsize=(12, 5), dpi=100)
for c, s in enumerate(steps):
    noisy = data if s == 0 else add_noise(data, torch.full((len(data),), s - 1), torch.randn_like(data))
    axes[0, c].scatter(noisy[:, 0], noisy[:, 1], s=1)
    axes[0, c].set_title(f"正向：第 {s} 步")
    back = hist[T - s]
    axes[1, c].scatter(back[:, 0], back[:, 1], s=1, color="#ef6c00")
    axes[1, c].set_title(f"反向：去噪 {s} 步")
for ax in axes.flat:
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
fig.tight_layout()
fig.savefig("lab9_3_diffusion.png")
final = hist[0]
def ring(p):
    return ((p.norm(dim=1) > 0.85) & (p.norm(dim=1) < 1.15)).float().mean()


inside, orig = ring(final), ring(data)
print(f"生成的 2000 个点中，落在“脸”的圆圈附近（离中心 0.85～1.15）的比例：{inside:.1%}（原始数据 {orig:.1%}）")
print("过程图已保存到 lab9_3_diffusion.png")
