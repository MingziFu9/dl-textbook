"""实验 1-2 准备 PyTorch 环境，运行现成程序识别手写数字

不用自己训练：加载随书提供的、已经训练好的网络（mnist_mlp.pt），
看它在 1 万张测试图片上认得准不准，再让它认一认你自己写的数字。
运行：python lab1_2_first_run.py            （认随书的示例图片 my_digit.png）
      python lab1_2_first_run.py 照片.jpg   （认你自己拍的数字照片）
"""
import platform
import sys

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import load_mnist, use_chinese_font  # noqa: E402

# ---------- 第 1 步：检查环境 ----------
print("① 环境检查")
print(f"  Python {platform.python_version()}，PyTorch {torch.__version__}")
print(f"  有没有可用的 GPU：{'有' if torch.cuda.is_available() else '没有，用 CPU 运行（本书的实验都够用）'}")

# ---------- 第 2 步：搭出网络的“骨架” ----------
model = nn.Sequential(
    nn.Flatten(),                 # 把 28×28 的图片摊平成 784 个数
    nn.Linear(784, 128), nn.ReLU(),   # 第 1 个隐藏层：128 个神经元
    nn.Linear(128, 64), nn.ReLU(),    # 第 2 个隐藏层：64 个神经元
    nn.Linear(64, 10),                # 输出层：10 个神经元，对应数字 0～9
)
print("\n② 网络结构")
print(model)
n_params = sum(p.numel() for p in model.parameters())
print(f"  参数（权重和偏置）一共 {n_params} 个")

# ---------- 第 3 步：装上训练好的参数，在测试集上考一考 ----------
model.load_state_dict(torch.load("mnist_mlp.pt"))
model.eval()                      # 切换到“使用”模式
x_test, y_test = load_mnist(train=False)
with torch.no_grad():             # 只做预测，不需要求梯度
    pred = model(x_test).argmax(dim=1)
acc = (pred == y_test).float().mean().item()
print(f"\n③ 在 {len(y_test)} 张测试图片上的准确率：{acc:.4f}，认错 {(pred != y_test).sum().item()} 张")

# 画出前 12 张测试图片和网络的回答
use_chinese_font()
fig, axes = plt.subplots(2, 6, figsize=(9, 3.4), dpi=120)
for i, ax in enumerate(axes.flat):
    ax.imshow(x_test[i, 0], cmap="gray_r")
    ok = "对" if pred[i] == y_test[i] else "错"
    ax.set_title(f"认作 {pred[i].item()}（{ok}）", fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig("lab1_2_test_images.png")
print("  前 12 张测试图片和识别结果已保存到 lab1_2_test_images.png")

# ---------- 第 4 步：认一认你自己写的数字 ----------
path = sys.argv[1] if len(sys.argv) > 1 else "my_digit.png"
img = Image.open(path).convert("L")           # 转成灰度图
img = ImageOps.invert(img)                    # MNIST 是黑底白字，纸上是白底黑字，所以要反色
img = img.crop(img.getbbox())                 # 裁掉四周空白
side = max(img.size) + 40                     # 补成正方形，四周留白，和 MNIST 的样子接近
canvas = Image.new("L", (side, side), 0)
canvas.paste(img, ((side - img.size[0]) // 2, (side - img.size[1]) // 2))
img = canvas.resize((28, 28), Image.BILINEAR)
x = torch.tensor(np.asarray(img), dtype=torch.float32).reshape(1, 1, 28, 28) / 255.0
with torch.no_grad():
    prob = torch.softmax(model(x), dim=1)[0]  # 把 10 个输出变成 10 个概率
top = prob.argsort(descending=True)[:3]
print(f"\n④ 识别 {path}")
for k in top:
    print(f"  数字 {k.item()}：概率 {prob[k].item():.3f}")
print(f"  网络的回答：{top[0].item()}")

# ---------- 第 5 步：用 PyTorch 验证 1.6 节的示范例题 ----------
tiny = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())   # 2-2-1 小网络
with torch.no_grad():                         # 把例题里的权重和偏置填进去
    tiny[0].weight[:] = torch.tensor([[0.5, -0.2], [-0.3, 0.4]])   # 第 1 行是隐藏神经元 h1 的两个权重
    tiny[0].bias[:] = torch.tensor([0.1, -0.1])
    tiny[2].weight[:] = torch.tensor([[1.0, -0.5]])
    tiny[2].bias[:] = torch.tensor([0.3])
    x = torch.tensor([[1.0, 2.0]])
    h = tiny[1](tiny[0](x))                   # 隐藏层的输出
    y = tiny(x)
print(f"\n⑤ 示范例题：隐藏层输出 h = {[round(v, 3) for v in h[0].tolist()]}，网络输出 y = {y.item():.3f}")
