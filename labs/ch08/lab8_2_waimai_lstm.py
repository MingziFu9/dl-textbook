"""实验 8-2 外卖评论情感分类（Embedding + LSTM）。

流程：分词 → 建词表、把词换成编号 → 补齐到同样长度 → 训练两个模型并比较：
① 词袋模型：把一条评论所有词的词向量取平均（不管顺序）；② LSTM：按顺序读词。
运行：python lab8_2_waimai_lstm.py（在 labs/ch08 目录下，约 2 分钟）
"""
import logging
import sys
import time
from collections import Counter

import jieba
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

sys.path.append("..")
from dlbook import load_waimai, set_seed  # noqa: E402

jieba.setLogLevel(logging.WARNING)
set_seed(42)

# ---------- 第 1 步：分词、划分数据 ----------
reviews, labels = load_waimai()
tokens = [jieba.lcut(r) for r in reviews]
perm = torch.randperm(len(reviews)).tolist()
idx_train, idx_val, idx_test = perm[:8000], perm[8000:10000], perm[10000:]
print(f"训练 {len(idx_train)} 条，验证 {len(idx_val)} 条，测试 {len(idx_test)} 条；"
      f"好评占 {sum(labels) / len(labels):.1%}")

# ---------- 第 2 步：建词表（只用训练集），把词换成编号 ----------
counter = Counter(w for i in idx_train for w in tokens[i])
vocab = {"<补齐>": 0, "<未知>": 1}
for w, c in counter.items():
    if c >= 2:                                    # 只保留在训练集中至少出现 2 次的词
        vocab[w] = len(vocab)
MAXLEN = 40
print(f"词表大小：{len(vocab)}；每条评论最多取 {MAXLEN} 个词")


def encode(words):
    ids = [vocab.get(w, 1) for w in words[:MAXLEN]]
    return ids + [0] * (MAXLEN - len(ids)), max(len(ids), 1)


def make(idxs):
    enc = [encode(tokens[i]) for i in idxs]
    return (torch.tensor([e[0] for e in enc]), torch.tensor([e[1] for e in enc]),
            torch.tensor([labels[i] for i in idxs]))


train, val, test = make(idx_train), make(idx_val), make(idx_test)
print("示例：", " / ".join(tokens[idx_train[0]]))
print("编号：", train[0][0].tolist()[:15], "……，真实长度", train[1][0].item())


# ---------- 第 3 步：两个模型 ----------
class BagModel(nn.Module):
    """词袋：所有词的词向量取平均，不管顺序。"""

    def __init__(self, v, dim=64):
        super().__init__()
        self.emb = nn.Embedding(v, dim, padding_idx=0)
        self.head = nn.Linear(dim, 2)

    def forward(self, x, lengths):
        e = self.emb(x)                                         # 批量 × 40 × 64
        return self.head(e.sum(1) / lengths.unsqueeze(1))       # 补齐的 0 号词向量恒为 0，不影响求和


class LSTMModel(nn.Module):
    """词向量 → LSTM 按顺序读 → 最后一个真实词处的隐藏状态 → 好评 / 差评。"""

    def __init__(self, v, dim=64, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(v, dim, padding_idx=0)
        self.lstm = nn.LSTM(dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)                           # 打包后，LSTM 读到真实的最后一个词就停
        return self.head(h[-1])


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    pred = model(data[0], data[1]).argmax(1)
    y = data[2]
    acc = (pred == y).float().mean().item()
    neg_recall = ((pred == 0) & (y == 0)).sum().item() / (y == 0).sum().item()   # 差评被找出来的比例
    neg_precision = ((pred == 0) & (y == 0)).sum().item() / max((pred == 0).sum().item(), 1)
    return acc, neg_precision, neg_recall


def fit(model, epochs=5, lr=0.002, name=""):
    set_seed(0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, start = -1, None, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        p = torch.randperm(len(train[2]))
        for i in range(0, len(p), 64):
            b = p[i:i + 64]
            loss = loss_fn(model(train[0][b], train[1][b]), train[2][b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc = evaluate(model, val)[0]
        print(f"  [{name}] 第 {epoch} 轮：验证准确率 {acc:.4f}")
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return time.time() - start


results = {}
for name, cls in [("① 词袋", BagModel), ("② LSTM", LSTMModel)]:
    set_seed(42)
    model = cls(len(vocab))
    print(f"\n{name}：参数 {sum(p.numel() for p in model.parameters())}")
    sec = fit(model, name=name)
    results[name] = (model, evaluate(model, test), sec)

print("\n测试集结果：")
for name, (_, (acc, p0, r0), sec) in results.items():
    print(f"  {name}：准确率 {acc:.4f}，差评的精确率 {p0:.3f}、召回率 {r0:.3f}，训练用时 {sec:.0f} 秒")

# ---------- 第 4 步：试几条自己写的评论 ----------
examples = ["好吃", "不好吃", "味道不错，就是送得太慢了", "送得太慢了，不过味道不错", "一般般吧"]
print("\n自己写的评论（好评概率）：")
for text in examples:
    ids, n = encode(jieba.lcut(text))
    x, lengths = torch.tensor([ids]), torch.tensor([n])
    probs = [results[k][0](x, lengths).softmax(1)[0, 1].item() for k in results]
    print(f"  {text:<16} 词袋 {probs[0]:.3f}   LSTM {probs[1]:.3f}")
