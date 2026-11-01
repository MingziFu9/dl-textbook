"""实验 9-2 训练一个小 GAN 生成手写数字。

生成器：100 个随机数 → 256 → 512 → 784 个像素（一张 28×28 的“假”数字）；
判别器：784 个像素 → 512 → 256 → 1 个数（这张图是真的概率）。
两者轮流训练 30 轮，每隔几轮保存一张生成器“画”的图，并画出两条损失曲线。
运行：python lab9_2_gan.py（在 labs/ch09 目录下，约 5 分钟）
"""
import sys
import time

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import load_mnist, plot_curves, set_seed, use_chinese_font  # noqa: E402

use_chinese_font()
set_seed(42)
X, _ = load_mnist(train=True)
X = X.flatten(1) * 2 - 1                  # 像素从 0～1 变到 -1～1，和生成器最后的 Tanh 对应
Z_DIM = 100

G = nn.Sequential(                        # 生成器（造假者）
    nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 784), nn.Tanh(),
)
D = nn.Sequential(                        # 判别器（鉴定师）
    nn.Linear(784, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 256), nn.LeakyReLU(0.2),
    nn.Linear(256, 1),                    # 输出一个分数，经过 Sigmoid 就是“是真图”的概率
)
print(f"生成器参数 {sum(p.numel() for p in G.parameters())}，判别器参数 {sum(p.numel() for p in D.parameters())}")

opt_g = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()              # 二分类交叉熵（自带 Sigmoid）
fixed_z = torch.randn(32, Z_DIM)          # 固定 32 组随机数，观察同一组输入画出的图怎样变化

history = {"判别器损失": [], "生成器损失": []}
snapshots = {}
start = time.time()
for epoch in range(1, 31):
    perm = torch.randperm(len(X))
    d_sum = g_sum = 0.0
    n = 0
    for i in range(0, len(X), 128):
        real = X[perm[i:i + 128]]
        b = len(real)
        ones, zeros = torch.ones(b, 1), torch.zeros(b, 1)

        # 第 1 步：训练判别器——真图判为 1，假图判为 0
        fake = G(torch.randn(b, Z_DIM))
        d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)   # detach：这一步不更新生成器
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # 第 2 步：训练生成器——让判别器把假图判为 1
        g_loss = bce(D(fake), ones)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        d_sum += d_loss.item()
        g_sum += g_loss.item()
        n += 1
    history["判别器损失"].append(d_sum / n)
    history["生成器损失"].append(g_sum / n)
    with torch.no_grad():
        p_real = torch.sigmoid(D(X[:1000])).mean().item()
        p_fake = torch.sigmoid(D(G(torch.randn(1000, Z_DIM)))).mean().item()
    if epoch in (1, 2, 5, 10, 20, 30):
        with torch.no_grad():
            snapshots[epoch] = G(fixed_z)
        print(f"  第 {epoch:2d} 轮：判别器损失 {d_sum / n:.3f}，生成器损失 {g_sum / n:.3f}，"
              f"判别器认为真图是真的概率 {p_real:.2f}、假图是真的概率 {p_fake:.2f}，已用 {time.time() - start:.0f} 秒")

plot_curves(history, "lab9_2_loss.png", "GAN 的两条损失曲线")

fig, axes = plt.subplots(len(snapshots), 8, figsize=(8, 6.4), dpi=110)
for r, (epoch, imgs) in enumerate(snapshots.items()):
    for c in range(8):
        axes[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray", vmin=-1, vmax=1)
        axes[r, c].axis("off")
    axes[r, 0].set_title(f"第 {epoch} 轮", fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("lab9_2_samples.png")
torch.save(G.state_dict(), "gan_generator.pt")
print("生成的图片已保存到 lab9_2_samples.png，生成器参数已保存到 gan_generator.pt")
