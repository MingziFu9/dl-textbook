# 第 8 章 文本分类实战与预训练模型：实验代码

对应教材：[第 8 章](https://mingzifu9.github.io/dl-textbook/chapters/08-text/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch08/lab8_1_word_vectors.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 8-1 | ★ 基础 | 30 分钟 | 用外卖评论训练 Word2Vec 词向量，找出相近的词，手算余弦相似度 | `lab8_1_word_vectors.py` |
| 8-2 | ★ 基础 | 45 分钟 | 外卖评论情感分类：分词、建词表、编号补齐，比较词袋模型和 LSTM | `lab8_2_waimai_lstm.py` |
| 8-3 | ☆ 拓展（待实测） | 45 分钟 | 用中文预训练模型 `hfl/rbt3` 微调同一任务，建议在云算力平台上运行 | `lab8_3_bert_finetune.py` |

实验 8-3 需要另外安装 `transformers`，并下载约 150 MB 的预训练模型，建议在云算力平台上运行，见附录 B。

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
