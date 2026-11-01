# 第 5 章 卷积神经网络：为图像而生：实验代码

对应教材：[第 5 章](https://mingzifu9.github.io/dl-textbook/chapters/05-cnn/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch05/lab5_1_conv_edge.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 5-1 | ★ 基础 | 30 分钟 | 手算卷积并用 `Conv2d` 验证；用三个卷积核处理一张照片，看特征图；观察步幅、填充、池化对尺寸的影响 | `lab5_1_conv_edge.py` |
| 5-2 | ★ 基础 | 45 分钟 | CIFAR-10 小图分类：同样的数据和训练方法，比较全连接网络与 CNN 的参数个数和准确率 | `lab5_2_mlp_vs_cnn.py` |
| 5-3 | ☆ 拓展 | 30 分钟 | 把 CNN 加深为三层卷积，再加批归一化，比较准确率和训练速度 | `lab5_3_deeper_bn.py` |

公共代码：`common05.py`（被本章实验程序导入，不用单独运行）。

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
