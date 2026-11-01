"""实验 7-3 分析认错的图片，写错误案例报告。

读取实验 7-2 微调好的 ResNet18，在测试集上：
1. 画混淆矩阵；2. 算每一类的精确率、召回率；3. 找出最常认错的类别对；4. 把认错的图片排成一张图，逐张分析原因。
运行：python lab7_3_error_analysis.py（在 labs/ch07 目录下，约 1 分钟；需先运行实验 7-2）
"""
import sys

import matplotlib
import torch
from torch import nn

from common07 import CLASSES, IMAGENET_NORM, X_TEST, Y_TEST, predict

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import use_chinese_font  # noqa: E402
from torchvision.models import resnet18  # noqa: E402

use_chinese_font()


class Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = resnet18()
        self.net.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.net(IMAGENET_NORM(x))


model = Pretrained()
model.load_state_dict(torch.load("imagenette_resnet18.pt", weights_only=True))
prob = predict(model, X_TEST).softmax(1)
conf, pred = prob.max(1)
print(f"测试集 {len(Y_TEST)} 张，认对 {(pred == Y_TEST).sum().item()} 张，准确率 {(pred == Y_TEST).float().mean():.4f}")

# ---------- 1. 混淆矩阵：第 i 行第 j 列 = 真实是 i 类、被认成 j 类的张数 ----------
cm = torch.zeros(10, 10, dtype=torch.long)
for t, p in zip(Y_TEST, pred):
    cm[t, p] += 1
fig, ax = plt.subplots(figsize=(7, 6), dpi=110)
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
ax.set_yticks(range(10), CLASSES)
ax.set_xlabel("预测类别")
ax.set_ylabel("真实类别")
for i in range(10):
    for j in range(10):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout()
fig.savefig("lab7_3_confusion.png")
print("混淆矩阵已保存到 lab7_3_confusion.png")

# ---------- 2. 每一类的精确率、召回率 ----------
print("\n类别        精确率   召回率   测试张数")
for c in range(10):
    precision = cm[c, c] / cm[:, c].sum()      # 认成 c 的里面，真是 c 的比例
    recall = cm[c, c] / cm[c].sum()            # 真是 c 的里面，认出来的比例
    print(f"  {CLASSES[c]:<6}  {precision:.3f}    {recall:.3f}    {int(cm[c].sum())}")

# ---------- 3. 最常认错的类别对 ----------
off = cm.clone()
off.fill_diagonal_(0)
print("\n最常认错的 5 对（真实 → 预测：张数）：")
for k in off.flatten().topk(5).indices:
    i, j = divmod(k.item(), 10)
    print(f"  {CLASSES[i]} → {CLASSES[j]}：{off[i, j].item()} 张")

# ---------- 4. 认错的图片：挑“最有把握却认错”的 12 张 ----------
wrong = (pred != Y_TEST).nonzero().flatten()
wrong = wrong[conf[wrong].argsort(descending=True)][:12]
fig, axes = plt.subplots(3, 4, figsize=(10, 8.4), dpi=100)
for ax, i in zip(axes.flat, wrong.tolist()):
    ax.imshow(X_TEST[i].permute(1, 2, 0))
    ax.set_title(f"#{i} 真：{CLASSES[Y_TEST[i]]}\n认成：{CLASSES[pred[i]]}（{conf[i]:.2f}）", fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("lab7_3_wrong.png")
print("\n最有把握却认错的 12 张已保存到 lab7_3_wrong.png：")
for i in wrong.tolist():
    print(f"  #{i}：真实 {CLASSES[Y_TEST[i]]}，认成 {CLASSES[pred[i]]}，把握 {conf[i]:.2f}")
