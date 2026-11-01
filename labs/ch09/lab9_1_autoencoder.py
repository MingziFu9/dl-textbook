"""实验 9-1 自编码器：压缩与还原手写数字，并给图片去噪。

1. 普通自编码器：784 个像素 → 32 个数 → 784 个像素，看还原效果；
2. 降噪自编码器：输入加了噪声的图片，要求还原出干净的图片；
3. 异常检测：用“还原误差”找出不是手写数字的图片（服装图片）。
运行：python lab9_1_autoencoder.py（在 labs/ch09 目录下，约 2 分钟）
"""
import sys
import time

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import DATA, load_mnist, set_seed, use_chinese_font  # noqa: E402

use_chinese_font()
X_TRAIN, _ = load_mnist(train=True)
X_TEST, _ = load_mnist(train=False)
X_TRAIN, X_TEST = X_TRAIN.flatten(1), X_TEST.flatten(1)          # 每张图摊平成 784 个数


def make_autoencoder(code=32):
    encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, code))          # 编码器：压缩
    decoder = nn.Sequential(nn.Linear(code, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())  # 解码器：还原
    return nn.Sequential(encoder, decoder)


def train(model, noise=0.0, epochs=5, name=""):
    set_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    start = time.time()
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(len(X_TRAIN))
        total = 0.0
        for i in range(0, len(perm), 128):
            clean = X_TRAIN[perm[i:i + 128]]
            noisy = (clean + noise * torch.randn_like(clean)).clamp(0, 1) if noise else clean
            loss = loss_fn(model(noisy), clean)      # 目标永远是干净的原图
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(clean)
        print(f"  [{name}] 第 {epoch} 轮：还原误差（均方误差）{total / len(X_TRAIN):.4f}，已用 {time.time() - start:.0f} 秒")


# ---------- 1. 普通自编码器 ----------
set_seed(42)
ae = make_autoencoder(32)
print(f"普通自编码器：784 → 32 → 784，参数 {sum(p.numel() for p in ae.parameters())}")
train(ae, name="普通")
with torch.no_grad():
    codes = ae[0](X_TEST[:1])
    recon = ae(X_TEST)
print("第 1 张测试图片压缩后的 32 个数（前 8 个）：", [round(v, 2) for v in codes[0, :8].tolist()])
print(f"测试集还原误差：{((recon - X_TEST) ** 2).mean():.4f}；压缩比 784 ÷ 32 = {784 / 32:.1f} 倍")

# ---------- 2. 降噪自编码器 ----------
set_seed(42)
dae = make_autoencoder(32)
train(dae, noise=0.5, name="降噪")
set_seed(1)
noisy_test = (X_TEST[:8] + 0.5 * torch.randn(8, 784)).clamp(0, 1)
with torch.no_grad():
    denoised = dae(noisy_test)
    plain = ae(noisy_test)
print(f"8 张加噪图片：降噪自编码器还原误差 {((denoised - X_TEST[:8]) ** 2).mean():.4f}，"
      f"普通自编码器 {((plain - X_TEST[:8]) ** 2).mean():.4f}，什么都不做 {((noisy_test - X_TEST[:8]) ** 2).mean():.4f}")

rows = [("原图", X_TEST[:8]), ("普通自编码器还原", recon[:8]), ("加噪声", noisy_test), ("降噪自编码器还原", denoised)]
fig, axes = plt.subplots(4, 8, figsize=(9, 5), dpi=110)
for r, (title, imgs) in enumerate(rows):
    for c in range(8):
        axes[r, c].imshow(imgs[c].reshape(28, 28), cmap="gray")
        axes[r, c].axis("off")
    axes[r, 0].set_title(title, fontsize=9, loc="left")
fig.tight_layout()
fig.savefig("lab9_1_reconstruct.png")
print("对比图已保存到 lab9_1_reconstruct.png")

# ---------- 3. 异常检测：还原误差 ----------
from torchvision import datasets  # noqa: E402

fashion = datasets.FashionMNIST(DATA, train=False, download=True).data[:1000].float().flatten(1) / 255
with torch.no_grad():
    err_digit = ((ae(X_TEST[:1000]) - X_TEST[:1000]) ** 2).mean(1)
    err_fashion = ((ae(fashion) - fashion) ** 2).mean(1)
threshold = err_digit.quantile(0.95)                 # 把 95% 的手写数字都能通过的误差作为门槛
print(f"\n还原误差平均值：手写数字 {err_digit.mean():.4f}，服装图片 {err_fashion.mean():.4f}")
print(f"门槛（手写数字误差的 95% 分位数）{threshold:.4f}：服装图片被判为“异常”的比例 {(err_fashion > threshold).float().mean():.1%}")
