"""实验 5-1 用 Conv2d 给图片做边缘检测，看特征图。

步骤：
1. 手算一个 5×5 图片与 3×3 卷积核的卷积，再用 Conv2d 验证；
2. 读一张照片，转成灰度图；
3. 用三个手工设置的卷积核（竖直边缘、水平边缘、模糊）做卷积，画出特征图；
4. 观察步幅、填充、池化对特征图尺寸的影响。
"""
import sys

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import load_imagenette, use_chinese_font  # noqa: E402

use_chinese_font()

# ---------- 第 1 步：手算与 Conv2d 对照 ----------
img5 = torch.tensor([[0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1],
                     [0, 0, 1, 1, 1]], dtype=torch.float32)
k_vertical = torch.tensor([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=torch.float32)
conv = nn.Conv2d(1, 1, kernel_size=3, bias=False)        # 1 个输入通道、1 个卷积核，3×3，不要偏置
with torch.no_grad():
    conv.weight[:] = k_vertical                           # 权重形状 1×1×3×3，直接填进去
out = conv(img5.reshape(1, 1, 5, 5))                      # Conv2d 要求输入形状：批量×通道×高×宽
print("第 1 步：5×5 图片与竖直边缘卷积核的卷积结果（3×3）：")
print(out.reshape(3, 3).detach())
print("左上角的值 =", float((img5[:3, :3] * k_vertical).sum()), "（左上 3×3 区域与卷积核对应相乘再相加）")

# ---------- 第 2 步：读一张照片 ----------
x, y, names = load_imagenette(train=False, n_per_class=10, size=160)
photo = x[y == 6][0]                                      # 第 6 类“垃圾车”的第 1 张
gray = (0.299 * photo[0] + 0.587 * photo[1] + 0.114 * photo[2]).reshape(1, 1, 160, 160)
print("\n第 2 步：照片（", names[6], "）灰度图形状：", tuple(gray.shape))

# ---------- 第 3 步：三个卷积核 ----------
kernels = {
    "竖直边缘": k_vertical,
    "水平边缘": k_vertical.T,
    "模糊（平均）": torch.full((3, 3), 1 / 9),
}
conv3 = nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False)   # 3 个卷积核 → 3 张特征图
with torch.no_grad():
    for i, k in enumerate(kernels.values()):
        conv3.weight[i, 0] = k
    maps = conv3(gray)
print("第 3 步：卷积层权重形状", tuple(conv3.weight.shape), "→ 输出特征图形状", tuple(maps.shape))

fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), dpi=120)
axes[0].imshow(gray[0, 0], cmap="gray")
axes[0].set_title("原图（灰度）")
for ax, (name, m) in zip(axes[1:], zip(kernels, maps[0])):
    ax.imshow(m.abs().detach() if "边缘" in name else m.detach(), cmap="gray")
    ax.set_title(name)
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig("lab5_1_feature_maps.png")
print("特征图已保存到 lab5_1_feature_maps.png")

# ---------- 第 4 步：尺寸变化 ----------
print("\n第 4 步：输入 1×1×160×160，看不同设置的输出尺寸")
settings = [
    ("3×3，步幅 1，不填充", nn.Conv2d(1, 1, 3)),
    ("3×3，步幅 1，填充 1", nn.Conv2d(1, 1, 3, padding=1)),
    ("3×3，步幅 2，填充 1", nn.Conv2d(1, 1, 3, stride=2, padding=1)),
    ("5×5，步幅 1，不填充", nn.Conv2d(1, 1, 5)),
    ("2×2 最大池化", nn.MaxPool2d(2)),
]
for name, layer in settings:
    print(f"  {name:<14} → {tuple(layer(gray).shape)}")

pool = nn.MaxPool2d(2)
small = torch.tensor([[1, 3, 2, 0], [4, 2, 1, 1], [0, 1, 5, 2], [2, 2, 3, 4]], dtype=torch.float32)
print("\n4×4 小例子做 2×2 最大池化：")
print(pool(small.reshape(1, 1, 4, 4)).reshape(2, 2))
