"""实验 7-2 迁移学习：用预训练 ResNet18 微调，和从零训练比。

三种做法用同样的数据（训练 2400 张、验证 600 张、测试 3925 张）：
① 从零训练：读取实验 7-1 保存的结果；
② 冻结特征提取部分，只训练新换上的最后一层；
③ 在 ② 的基础上，解冻全部参数，用很小的学习率再微调几轮。
运行：python lab7_2_transfer.py（在 labs/ch07 目录下，约 10 分钟；需先运行实验 7-1）
"""
import sys

import torch
from torch import nn

from common07 import IMAGENET_NORM, X_TEST, Y_TEST, accuracy, fit

sys.path.append("..")
from dlbook import load_pretrained_resnet18, set_seed  # noqa: E402

# ---------- ① 从零训练（实验 7-1 的结果） ----------
scratch_acc = torch.load("imagenette_cnn.pt", weights_only=True)["test_acc"]
print(f"① 从零训练（实验 7-1）：测试准确率 {scratch_acc:.4f}")


# ---------- 准备预训练模型 ----------
class Pretrained(nn.Module):
    """先按 ImageNet 的要求标准化，再送进 ResNet18。"""

    def __init__(self):
        super().__init__()
        self.net = load_pretrained_resnet18()
        print(f"  预训练 ResNet18 参数：{sum(p.numel() for p in self.net.parameters())}，"
              f"原来的最后一层：{self.net.fc}")
        self.net.fc = nn.Linear(512, 10)          # 换成新的最后一层：输出 10 类

    def forward(self, x):
        return self.net(IMAGENET_NORM(x))


set_seed(42)
model = Pretrained()

# ---------- ② 冻结：只训练最后一层 ----------
for p in model.net.parameters():
    p.requires_grad = False                       # 冻结全部参数
for p in model.net.fc.parameters():
    p.requires_grad = True                        # 只有新的最后一层参与训练
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n② 冻结特征提取部分：要训练的参数 {n_train} 个")
t2 = fit(model, model.net.fc.parameters(), epochs=5, lr=0.001, name="冻结")
acc2 = accuracy(model, X_TEST, Y_TEST)
print(f"② 冻结、只训练最后一层：测试准确率 {acc2:.4f}")

# ---------- ③ 微调：解冻全部，小学习率 ----------
for p in model.net.parameters():
    p.requires_grad = True
print(f"\n③ 解冻全部参数微调：要训练的参数 {sum(p.numel() for p in model.parameters())} 个")
t3 = fit(model, model.parameters(), epochs=3, lr=0.0001, name="微调")
acc3 = accuracy(model, X_TEST, Y_TEST)
print(f"③ 解冻微调：测试准确率 {acc3:.4f}")
torch.save(model.state_dict(), "imagenette_resnet18.pt")
print("微调后的模型已保存到 imagenette_resnet18.pt（实验 7-3 要用）")

print("\n汇总：")
print(f"  ① 从零训练            测试准确率 {scratch_acc:.4f}")
print(f"  ② 冻结，只训练最后一层  测试准确率 {acc2:.4f}  训练用时 {t2:.0f} 秒")
print(f"  ③ 再解冻全部微调 3 轮   测试准确率 {acc3:.4f}  训练用时 {t3:.0f} 秒")
