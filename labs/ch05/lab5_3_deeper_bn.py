"""实验 5-3 把 CNN 加深一层、加批归一化，看准确率变化。

三个网络：① 实验 5-2 的两层 CNN；② 加深为三层卷积；③ 三层卷积 + 批归一化（BatchNorm2d）。
数据、优化器、轮数都和实验 5-2 相同。
运行：python lab5_3_deeper_bn.py（在 labs/ch05 目录下，约 3 分钟）
"""
import sys

from torch import nn

from common05 import count_params, train

sys.path.append("..")
from dlbook import plot_curves, set_seed  # noqa: E402


def block(c_in, c_out, bn=False):
    """一个“卷积块”：卷积 →（批归一化）→ ReLU → 2×2 最大池化，高和宽各减半。"""
    layers = [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)]
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    layers += [nn.ReLU(), nn.MaxPool2d(2)]
    return layers


def make_cnn(channels, bn=False):
    layers, c_in = [], 3
    for c in channels:
        layers += block(c_in, c, bn)
        c_in = c
    size = 32 // 2 ** len(channels)                  # 每个卷积块把边长减半
    layers += [nn.Flatten(), nn.Linear(c_in * size * size, 128), nn.ReLU(), nn.Linear(128, 10)]
    return nn.Sequential(*layers)


results = {}
for name, channels, bn in [("两层卷积", [16, 32], False),
                           ("三层卷积", [16, 32, 64], False),
                           ("三层卷积+BN", [16, 32, 64], True)]:
    set_seed(42)
    model = make_cnn(channels, bn)
    print(f"\n{name}：参数 {count_params(model)}")
    accs, sec = train(model, epochs=10, name=name)
    results[name] = (count_params(model), accs, sec)

print("\n汇总：")
for name, (n, accs, sec) in results.items():
    print(f"  {name:<8} 参数 {n:>6}  第 5 轮 {accs[4]:.4f}  第 10 轮 {accs[-1]:.4f}  最好 {max(accs):.4f}  用时 {sec:.0f} 秒")
plot_curves({k: v[1] for k, v in results.items()}, "lab5_3_acc.png", "测试准确率")
