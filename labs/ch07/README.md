# 第 7 章 图像分类实战与迁移学习：实验代码

对应教材：[第 7 章](https://mingzifu9.github.io/dl-textbook/chapters/07-image-transfer/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch07/lab7_1_image_project.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 7-1 | ★ 基础 | 45 分钟 | 10 类图片分类完整项目：读数据、划分、从零训练 CNN（数据增强 + 早停）、评估、保存、读回预测 | `lab7_1_image_project.py` |
| 7-2 | ★ 基础 | 40 分钟 | 迁移学习：加载预训练 ResNet18，先冻结训练、再解冻微调，与从零训练比较；补充程序换用 CIFAR-10 验证 | `lab7_2_extra_cifar.py`<br>`lab7_2_transfer.py` |
| 7-3 | ☆ 拓展 | 40 分钟 | 画混淆矩阵，算精确率、召回率，找出最常认错的类别，分析认错的图片，写错误案例报告 | `lab7_3_error_analysis.py` |

公共代码：`common07.py`（被本章实验程序导入，不用单独运行）。

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
