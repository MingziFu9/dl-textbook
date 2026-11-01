# 第 3 章 反向传播：误差怎么往回传：实验代码

对应教材：[第 3 章](https://mingzifu9.github.io/dl-textbook/chapters/03-backprop/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch03/lab3_1_autograd.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 3-1 | ★ 基础 | 45 分钟 | 用自动求导求两步计算、一个神经元、2-2-1 小网络的梯度，与手算和数值差分对照 | `lab3_1_autograd.py` |
| 3-2 | ☆ 拓展 | 30 分钟 | 10 层网络中测量每层的梯度，比较 Sigmoid、ReLU、初始化、批归一化；比较 SGD、动量、Adam 三种优化器 | `lab3_2_vanishing_optim.py` |

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
