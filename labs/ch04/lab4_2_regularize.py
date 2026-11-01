"""实验 4-2 加 Dropout、权重衰减、数据增强和早停，比一比

和实验 4-1 相同的网络和数据，分别加上不同的“防过拟合”办法，各训练最多 60 轮，
比较最后一轮的训练准确率、验证损失、验证准确率和测试准确率。
运行：python lab4_2_regularize.py（在 labs/ch04 目录下，约 1 分钟）
"""
import sys

from torchvision.transforms import v2

from common04 import X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, Y_VAL, evaluate, fit, make_mlp, set_seed

sys.path.append("..")
from dlbook import plot_curves  # noqa: E402

# 数据增强：每批图片随机旋转 ±10 度、上下左右平移最多 10%
augment = v2.RandomAffine(degrees=10, translate=(0.1, 0.1))

settings = [
    ("① 什么都不加", {}, {}),
    ("② Dropout 0.5", {"dropout": 0.5}, {}),
    ("③ 权重衰减 1e-3", {}, {"weight_decay": 1e-3}),
    ("④ 数据增强", {}, {"augment": augment}),
    ("⑤ 早停（耐心 5 轮）", {}, {"patience": 5}),
    ("⑥ Dropout + 数据增强 + 早停", {"dropout": 0.5}, {"augment": augment, "patience": 10}),
]
curves = {}
print("做法                         训练轮数  训练准确率  验证损失  验证准确率  测试准确率")
for name, model_kw, fit_kw in settings:
    set_seed(42)
    model = make_mlp(hidden=512, **model_kw)
    hist = fit(model, epochs=60, **fit_kw)
    _, train_acc = evaluate(model, X_TRAIN, Y_TRAIN)
    vl, va = evaluate(model, X_VAL, Y_VAL)
    _, test_acc = evaluate(model, X_TEST, Y_TEST)
    print(f"{name:<24}{len(hist['val_loss']):>6}{train_acc:>11.3f}{vl:>10.4f}{va:>11.3f}{test_acc:>11.4f}")
    curves[name[:1] + " 验证损失"] = hist["val_loss"]
plot_curves({k: v for k, v in curves.items() if k[0] in "①②④"}, "lab4_2_val_loss.png",
            title="三种做法的验证损失")
