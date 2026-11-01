# 第 6 章 序列模型：从 RNN 到 Transformer：实验代码

对应教材：[第 6 章](https://mingzifu9.github.io/dl-textbook/chapters/06-sequence/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch06/lab6_1_lstm_temperature.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 6-1 | ★ 基础 | 40 分钟 | 用前 30 天的气温预测第 31 天，比较“明天和今天一样”、简单 RNN、LSTM | `lab6_1_lstm_temperature.py` |
| 6-2 | ★ 基础 | 40 分钟 | 用 `nn.MultiheadAttention` 核对手算的注意力；训练一个外卖评论注意力分类器，画注意力热力图 | `lab6_2_attention_heatmap.py` |
| 6-3 | ☆ 拓展 | 25 分钟 | 同一气温预测任务换成 Transformer 编码器，与 LSTM 比较误差和速度 | `lab6_3_transformer.py` |

公共代码：`common06.py`（被本章实验程序导入，不用单独运行）。

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
