"""实验 7-1 10 类图片分类完整项目（从零训练一个 CNN）。

流程：读数据并划分 → 看几张样例 → 搭网络 → 训练（数据增强 + 早停）→ 在测试集上评估 → 保存模型 → 读回模型预测一张新图片。
运行：python lab7_1_image_project.py（在 labs/ch07 目录下，约 5 分钟）
"""
import sys

import matplotlib
import torch
from torch import nn

from common07 import CLASSES, SIZE, X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, accuracy, fit, predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import set_seed, use_chinese_font  # noqa: E402

# ---------- 第 1 步：看看数据 ----------
print("类别：", "、".join(CLASSES))
print(f"训练集 {tuple(X_TRAIN.shape)}，验证集 {tuple(X_VAL.shape)}，测试集 {tuple(X_TEST.shape)}")
print("训练集每类张数：", torch.bincount(Y_TRAIN).tolist())
use_chinese_font()
fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), dpi=110)
for c, ax in enumerate(axes.flat):
    ax.imshow(X_TRAIN[Y_TRAIN == c][0].permute(1, 2, 0))       # 通道×高×宽 → 高×宽×通道 才能显示
    ax.set_title(CLASSES[c])
    ax.axis("off")
fig.tight_layout()
fig.savefig("lab7_1_samples.png")
print("样例图片已保存到 lab7_1_samples.png")


# ---------- 第 2 步：搭网络 ----------
def block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]


set_seed(42)
model = nn.Sequential(
    *block(3, 32), *block(32, 64), *block(64, 128), *block(128, 256),   # 112 → 56 → 28 → 14 → 7
    nn.AdaptiveAvgPool2d(1),                                            # 全局平均池化：每张特征图平均成 1 个数
    nn.Flatten(),
    nn.Dropout(0.3),
    nn.Linear(256, 10),
)
print(f"\n网络参数：{sum(p.numel() for p in model.parameters())}")

# ---------- 第 3 步：训练 ----------
fit(model, model.parameters(), epochs=30, lr=0.001, name="从零训练", patience=6)

# ---------- 第 4 步：在测试集上评估（只做这一次） ----------
test_acc = accuracy(model, X_TEST, Y_TEST)
print(f"\n测试集（{len(X_TEST)} 张）准确率：{test_acc:.4f}")

# ---------- 第 5 步：保存模型 ----------
torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "size": SIZE, "test_acc": test_acc},
           "imagenette_cnn.pt")
print("模型已保存到 imagenette_cnn.pt")

# ---------- 第 6 步：读回模型，预测一张“新”图片 ----------
saved = torch.load("imagenette_cnn.pt", weights_only=True)
model.load_state_dict(saved["state_dict"])
img = X_TEST[1234]
prob = predict(model, img[None]).softmax(1)[0]
top3 = prob.topk(3)
print(f"\n测试集第 1234 张，真实类别：{CLASSES[Y_TEST[1234]]}")
for p, c in zip(top3.values, top3.indices):
    print(f"  {saved['classes'][c]}：{p:.3f}")
