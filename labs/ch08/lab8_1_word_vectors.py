"""实验 8-1 训练一个小词向量，看哪些词挨得近。

用约 1.2 万条外卖评论：先用 jieba 分词，再用 gensim 的 Word2Vec 训练 50 维词向量，
查看和某个词最相近的词，并手算两个词的余弦相似度，与 gensim 的结果核对。
运行：python lab8_1_word_vectors.py（在 labs/ch08 目录下，约 30 秒）
"""
import logging
import sys

import jieba
import numpy as np
from gensim.models import Word2Vec

sys.path.append("..")
from dlbook import load_waimai  # noqa: E402

jieba.setLogLevel(logging.WARNING)

# ---------- 第 1 步：分词 ----------
reviews, labels = load_waimai()
sentences = [jieba.lcut(r) for r in reviews]
print("分词示例：")
for s in sentences[:3]:
    print("  " + " / ".join(s))
n_words = sum(len(s) for s in sentences)
print(f"共 {len(sentences)} 条评论，{n_words} 个词")

# ---------- 第 2 步：训练词向量 ----------
model = Word2Vec(sentences, vector_size=50, window=5, min_count=5, sg=1, epochs=10, seed=42, workers=1)
wv = model.wv
print(f"\n词表大小（出现至少 5 次的词）：{len(wv)}；每个词是 {wv.vector_size} 维向量")
print("“好吃”的词向量前 5 个数：", np.round(wv["好吃"][:5], 3))

# ---------- 第 3 步：找最相近的词 ----------
for word in ["好吃", "难吃", "快", "慢", "米饭"]:
    sims = wv.most_similar(word, topn=5)
    print(f"和“{word}”最相近：" + "、".join(f"{w}({s:.2f})" for w, s in sims))

# ---------- 第 4 步：手算余弦相似度 ----------
a, b = wv["好吃"], wv["难吃"]
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"\n手算 cos(好吃, 难吃) = {cos:.4f}；gensim 计算 = {wv.similarity('好吃', '难吃'):.4f}")
for w1, w2 in [("好吃", "美味"), ("快", "及时"), ("好吃", "外卖员")]:
    if w1 in wv and w2 in wv:
        print(f"  相似度({w1}, {w2}) = {wv.similarity(w1, w2):.4f}")
