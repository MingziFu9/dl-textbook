"""实验 6-2 用现成的多头注意力层，画一句话的注意力热力图。

第 1 部分：3 个词、每个词 2 个数，手算注意力，再用 nn.MultiheadAttention 核对。
第 2 部分：用外卖评论训练一个小小的“注意力分类器”（好评 / 差评），画出它读评论时“看哪里”。
运行：python lab6_2_attention_heatmap.py（在 labs/ch06 目录下，约 1 分钟）
"""
import sys

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append("..")
from dlbook import load_waimai, set_seed, use_chinese_font  # noqa: E402

torch.set_printoptions(precision=3, sci_mode=False)

# ================= 第 1 部分：手算注意力，再用 PyTorch 核对 =================
print("第 1 部分：手算注意力")
X = torch.tensor([[1.0, 0.0],     # 词 A
                  [0.0, 1.0],     # 词 B
                  [1.0, 1.0]])    # 词 C
scores = X @ X.T / 2 ** 0.5       # 每两个词的向量做点积，再除以 √2（向量长度 2 的平方根）
weights = scores.softmax(dim=-1)  # 每一行变成加起来等于 1 的注意力权重
out = weights @ X                 # 按权重把所有词的向量加权平均
print("打分（点积 ÷ √2）：\n", scores)
print("注意力权重（每行加起来等于 1）：\n", weights)
print("输出：\n", out)

mha = nn.MultiheadAttention(embed_dim=2, num_heads=1, bias=False, batch_first=True)
with torch.no_grad():                                 # 把内部的变换都设成“原样不变”，只留注意力本身
    mha.in_proj_weight[:] = torch.cat([torch.eye(2)] * 3)
    mha.out_proj.weight[:] = torch.eye(2)
out2, w2 = mha(X[None], X[None], X[None])             # 查询、键、值都是同一句话：自注意力
print("nn.MultiheadAttention 的注意力权重：\n", w2[0].detach())
print("两种算法的输出最大相差：", float((out2[0].detach() - out).abs().max()))

# ================= 第 2 部分：训练一个注意力分类器 =================
print("\n第 2 部分：外卖评论好评 / 差评分类")
set_seed(42)
reviews, labels = load_waimai()
MAXLEN = 40
chars = sorted({c for r in reviews for c in r})
vocab = {c: i + 2 for i, c in enumerate(chars)}      # 0 留给“补齐”，1 留给“没见过的字”


def encode(text):
    ids = [vocab.get(c, 1) for c in text[:MAXLEN]]
    return ids + [0] * (MAXLEN - len(ids))


perm = torch.randperm(len(reviews)).tolist()
x_all = torch.tensor([encode(reviews[i]) for i in perm])
y_all = torch.tensor([labels[i] for i in perm])
n_test = 2000
x_test, y_test, x_train, y_train = x_all[:n_test], y_all[:n_test], x_all[n_test:], y_all[n_test:]
print(f"评论 {len(reviews)} 条（训练 {len(x_train)}，测试 {len(x_test)}），不同的字 {len(chars)} 个，每条最多取 {MAXLEN} 个字")


class AttnClassifier(nn.Module):
    """字向量 + 位置向量 → 多头自注意力 → 平均 → 好评 / 差评。"""

    def __init__(self, vocab_size, dim=64, heads=4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos = nn.Embedding(MAXLEN, dim)                     # 位置向量：告诉网络每个字在第几位
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Linear(dim, 2)

    def forward(self, x, return_attn=False):
        pad = x == 0                                             # 补齐的位置不参与注意力
        h = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        a, w = self.attn(h, h, h, key_padding_mask=pad)          # w：批量 × 字数 × 字数，4 个头的平均
        h = h + a                                                # 残差连接
        keep = (~pad).unsqueeze(-1).float()
        pooled = (h * keep).sum(1) / keep.sum(1)                 # 对真实的字取平均
        logits = self.head(pooled)
        return (logits, w) if return_attn else logits


model = AttnClassifier(len(vocab) + 2)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
for epoch in range(1, 6):
    model.train()
    p = torch.randperm(len(x_train))
    for i in range(0, len(x_train), 64):
        idx = p[i:i + 64]
        loss = loss_fn(model(x_train[idx]), y_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(x_test).argmax(1) == y_test).float().mean()
    print(f"  第 {epoch} 轮：测试准确率 {acc:.4f}")

# ---------- 画热力图 ----------
use_chinese_font()
examples = ["送餐很快，味道也不错", "等了两个小时，饭都凉了", "菜很好吃就是太贵了"]
fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), dpi=120)
model.eval()
for ax, text in zip(axes, examples):
    with torch.no_grad():
        logits, w = model(torch.tensor([encode(text)]), return_attn=True)
    n = len(text)
    m = w[0, :n, :n].numpy()
    ax.imshow(m, cmap="Blues")
    ax.set_xticks(range(n), list(text))
    ax.set_yticks(range(n), list(text))
    prob = logits.softmax(-1)[0, 1].item()
    ax.set_title(f"{text}\n好评概率 {prob:.2f}", fontsize=10)
    ax.set_xlabel("被看的字")
    received = m.mean(0)                                         # 每个字平均“被看”了多少
    top = sorted(range(n), key=lambda j: -received[j])[:3]
    print(f"\n“{text}”：好评概率 {prob:.3f}；最受关注的字：{'、'.join(text[j] + f'({received[j]:.2f})' for j in top)}")
axes[0].set_ylabel("正在读的字")
fig.tight_layout()
fig.savefig("lab6_2_heatmap.png")
print("\n热力图已保存到 lab6_2_heatmap.png")
