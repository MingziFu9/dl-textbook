"""实验 5-2 CIFAR-10 小图分类：全连接网络与 CNN 比一比。

两个网络用同样的数据（训练 2 万张、测试 1 万张）、同样的优化器和轮数，比较参数个数、准确率和用时。
运行：python lab5_2_mlp_vs_cnn.py（在 labs/ch05 目录下，约 3 分钟）
"""
import sys

import torch
from torch import nn

from common05 import CIFAR10_CN, X_TEST, X_TRAIN, Y_TEST, count_params, evaluate, train

sys.path.append("..")
from dlbook import plot_curves, set_seed  # noqa: E402

print("训练集", tuple(X_TRAIN.shape), "测试集", tuple(X_TEST.shape))

# ---------- 网络一：全连接网络（把 3×32×32 的图片摊平成 3072 个数） ----------
set_seed(42)
mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(3 * 32 * 32, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

# ---------- 网络二：卷积神经网络 ----------
set_seed(42)
cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),   # 3×32×32 → 16×32×32
    nn.MaxPool2d(2),                                         # → 16×16×16
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),  # → 32×16×16
    nn.MaxPool2d(2),                                         # → 32×8×8
    nn.Flatten(),                                            # → 2048
    nn.Linear(32 * 8 * 8, 128), nn.ReLU(),
    nn.Linear(128, 10),
)

print("\n逐层查看 CNN 的输出形状：")
out = X_TRAIN[:1]
for layer in cnn:
    out = layer(out)
    print(f"  {layer.__class__.__name__:<10} → {tuple(out.shape)}")

print(f"\n全连接网络参数：{count_params(mlp)}，CNN 参数：{count_params(cnn)}")
print("CNN 各层参数：", [p.numel() for p in cnn.parameters()])

mlp_accs, mlp_time = train(mlp, epochs=10, name="全连接")
cnn_accs, cnn_time = train(cnn, epochs=10, name="CNN")

print(f"\n全连接网络：参数 {count_params(mlp)}，最终测试准确率 {mlp_accs[-1]:.4f}，用时 {mlp_time:.0f} 秒")
print(f"CNN       ：参数 {count_params(cnn)}，最终测试准确率 {cnn_accs[-1]:.4f}，用时 {cnn_time:.0f} 秒")
plot_curves({"全连接网络": mlp_accs, "CNN": cnn_accs}, "lab5_2_acc.png", "测试准确率")

# 每一类的准确率
pred = torch.cat([cnn(X_TEST[i:i + 1000]).argmax(1) for i in range(0, 10000, 1000)])
print("\nCNN 每一类的测试准确率：")
for c in range(10):
    mask = Y_TEST == c
    print(f"  {CIFAR10_CN[c]}：{(pred[mask] == c).float().mean():.3f}", end="")
print()
torch.save(cnn.state_dict(), "cifar_cnn.pt")
print("CNN 参数已保存到 cifar_cnn.pt")
