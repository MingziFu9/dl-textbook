"""实验 4-1 故意制造过拟合：用 1000 张图片训练，读懂损失曲线

一个参数很多的网络（784 → 512 → 512 → 10，约 67 万个参数）只用 1000 张训练图片，训练 60 轮，
每轮记录训练集和验证集的损失、准确率，画出曲线，找出验证损失最低的那一轮。
运行：python lab4_1_overfit.py（在 labs/ch04 目录下，约 20 秒）
"""
import sys

from common04 import X_TEST, Y_TEST, evaluate, fit, make_mlp, set_seed

sys.path.append("..")
from dlbook import plot_curves  # noqa: E402

set_seed(42)
model = make_mlp(hidden=512)
print(f"参数个数：{sum(p.numel() for p in model.parameters())}，训练图片：1000 张")
hist = fit(model, epochs=60, verbose=True)
be = hist["best_epoch"]
print(f"\n验证损失最低在第 {be} 轮：{hist['val_loss'][be - 1]:.4f}（验证准确率 {hist['val_acc'][be - 1]:.3f}）")
print(f"第 60 轮：训练损失 {hist['train_loss'][-1]:.5f}，验证损失 {hist['val_loss'][-1]:.4f}，"
      f"两者相差 {hist['val_loss'][-1] / hist['train_loss'][-1]:.0f} 倍")
tl, ta = evaluate(model, X_TEST, Y_TEST)
print(f"第 60 轮的模型在测试集上：损失 {tl:.4f}，准确率 {ta:.4f}")
plot_curves({"训练损失": hist["train_loss"], "验证损失": hist["val_loss"]}, "lab4_1_loss.png",
            title="1000 张训练图片：训练损失与验证损失")
