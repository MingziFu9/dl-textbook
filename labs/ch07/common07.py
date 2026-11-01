"""第 7 章三个实验共用的数据、训练和评估函数。

数据：Imagenette（ImageNet 中挑出的 10 类照片），缩放到 SIZE×SIZE。
- 训练集：从官方训练文件夹每类取 300 张，其中 240 张训练、60 张作验证集（共 2400 + 600 张）；
- 测试集：官方验证文件夹的全部 3925 张，只在最后评估一次。
"""
import sys
import time

import torch
from torch import nn
from torchvision.transforms import v2

sys.path.append("..")
from dlbook import load_imagenette, set_seed  # noqa: E402

SIZE = 112
set_seed(42)
_x, _y, CLASSES = load_imagenette(train=True, n_per_class=300, size=SIZE)
_perm = torch.randperm(len(_x))
X_TRAIN, Y_TRAIN = _x[_perm[:2400]], _y[_perm[:2400]]
X_VAL, Y_VAL = _x[_perm[2400:]], _y[_perm[2400:]]
X_TEST, Y_TEST, _ = load_imagenette(train=False, size=SIZE)

# 训练时的数据增强：随机裁剪一部分再放大回原尺寸、随机水平翻转
AUGMENT = v2.Compose([v2.RandomResizedCrop(SIZE, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip()])
# 预训练模型要求的标准化（ImageNet 训练时用的平均值和标准差）
IMAGENET_NORM = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


@torch.no_grad()
def predict(model, x, batch_size=200):
    model.eval()
    return torch.cat([model(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def accuracy(model, x, y):
    return (predict(model, x).argmax(1) == y).float().mean().item()


def fit(model, params, epochs, lr, augment=True, batch_size=32, name="模型", patience=None):
    """训练并在每轮后看验证集；patience 不为空时做早停，并恢复验证准确率最好的那一轮参数。"""
    set_seed(0)
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch, wait = -1, None, 0, 0
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_TRAIN))
        total = 0.0
        for i in range(0, len(X_TRAIN), batch_size):
            idx = perm[i:i + batch_size]
            xb = X_TRAIN[idx]
            if augment:
                xb = torch.stack([AUGMENT(img) for img in xb])
            loss = loss_fn(model(xb), Y_TRAIN[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_acc = accuracy(model, X_VAL, Y_VAL)
        print(f"  [{name}] 第 {epoch:2d} 轮：训练损失 {total / len(X_TRAIN):.4f}，验证准确率 {val_acc:.4f}，"
              f"已用 {time.time() - start:.0f} 秒")
        if val_acc > best:
            best, best_epoch, wait = val_acc, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if patience and wait >= patience:
                print(f"  验证准确率连续 {patience} 轮没有提高，提前停止")
                break
    model.load_state_dict(best_state)
    print(f"  [{name}] 验证准确率最好在第 {best_epoch} 轮：{best:.4f}，共用时 {time.time() - start:.0f} 秒")
    return time.time() - start
