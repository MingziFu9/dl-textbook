# 第 4 章 把模型训练好：过拟合与调优：实验代码

对应教材：[第 4 章](https://mingzifu9.github.io/dl-textbook/chapters/04-generalization/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch04/lab4_1_overfit.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 4-1 | ★ 基础 | 30 分钟 | 用 1000 张图片训练一个大网络，画出训练与验证损失曲线，诊断过拟合 | `lab4_1_overfit.py` |
| 4-2 | ★ 基础 | 45 分钟 | 分别加 Dropout、权重衰减、数据增强、早停，比较它们对过拟合的效果 | `lab4_2_regularize.py` |
| 4-3 | ☆ 拓展 | 30 分钟 | 12 组超参数的网格搜索，只用验证集选择，最后在测试集上考一次 | `lab4_3_grid.py` |

公共代码：`common04.py`（被本章实验程序导入，不用单独运行）。

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
