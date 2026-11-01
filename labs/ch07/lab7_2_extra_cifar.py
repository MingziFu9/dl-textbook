"""实验 7-2 补充：换一个“不是来自 ImageNet”的数据，迁移学习还有效吗？

Imagenette 的 10 类照片本来就是从 ImageNet 里挑出来的，预训练模型等于“见过”这些类别，提升会偏大。
这里换成 CIFAR-10（另外收集的 32×32 小图），每类只取 100 张训练，比较从零训练和冻结预训练 ResNet18。
运行：python lab7_2_extra_cifar.py（在 labs/ch07 目录下，约 3 分钟）
"""
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("..")
from dlbook import load_cifar10, load_pretrained_resnet18, set_seed  # noqa: E402

set_seed(42)
x_train, y_train = load_cifar10(train=True, n_per_class=100)      # 1000 张
x_test, y_test = load_cifar10(train=False, n_per_class=200)       # 2000 张
print(f"训练 {len(x_train)} 张，测试 {len(x_test)} 张")
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def train_eval(model, params, x_tr, x_te, epochs, lr):
    set_seed(0)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 32):
            idx = perm[i:i + 32]
            loss = F.cross_entropy(model(x_tr[idx]), y_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return (model(x_te).argmax(1) == y_test).float().mean().item()


# ① 从零训练：第 5 章实验 5-3 中最好的“三层卷积 + 批归一化”
def block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(), nn.MaxPool2d(2)]


set_seed(42)
cnn = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64), nn.Flatten(),
                    nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10))
start = time.time()
acc1 = train_eval(cnn, cnn.parameters(), x_train, x_test, epochs=20, lr=0.001)
print(f"① 从零训练（20 轮）：测试准确率 {acc1:.4f}，用时 {time.time() - start:.0f} 秒")

# ② 冻结预训练 ResNet18：先把所有图片放大到 112×112、提取 512 维特征，再只训练最后一层
net = load_pretrained_resnet18()
net.fc = nn.Identity()                        # 去掉最后一层，输出 512 维特征
net.eval()
start = time.time()
with torch.no_grad():
    def features(x):
        x = F.interpolate(x, size=112, mode="bilinear", align_corners=False)
        return torch.cat([net((x[i:i + 200] - MEAN) / STD) for i in range(0, len(x), 200)])
    f_train, f_test = features(x_train), features(x_test)
head = nn.Linear(512, 10)
acc2 = train_eval(head, head.parameters(), f_train, f_test, epochs=30, lr=0.001)
print(f"② 冻结预训练 ResNet18，只训练最后一层：测试准确率 {acc2:.4f}，用时 {time.time() - start:.0f} 秒")
