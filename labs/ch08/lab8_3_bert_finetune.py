"""实验 8-3（拓展，待实测）用中文预训练模型微调外卖评论分类。

说明：编者的实验环境无法访问模型下载网站，本程序尚未完整运行，运行结果请以你自己的为准，
欢迎把结果反馈给编者补进教材。建议在有 GPU 的云算力平台（如百度 AI Studio、阿里云 PAI、魔搭 Notebook）上运行。

需要额外安装：pip install transformers
模型：hfl/rbt3（哈工大讯飞联合实验室发布的 3 层中文 RoBERTa，约 3800 万参数，比 BERT-base 小、快）。
国内下载慢时，运行前设置镜像：Linux / macOS 执行 export HF_ENDPOINT=https://hf-mirror.com，
Windows 执行 set HF_ENDPOINT=https://hf-mirror.com。
运行：python lab8_3_bert_finetune.py（在 labs/ch08 目录下；GPU 上预计几分钟，CPU 上可能要半小时以上）
"""
import sys
import time

import torch

sys.path.append("..")
from dlbook import load_waimai, set_seed  # noqa: E402

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

MODEL = "hfl/rbt3"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)
set_seed(42)

# ---------- 第 1 步：数据，划分方法与实验 8-2 相同 ----------
reviews, labels = load_waimai()
perm = torch.randperm(len(reviews)).tolist()
idx_train, idx_val, idx_test = perm[:8000], perm[8000:10000], perm[10000:]

# ---------- 第 2 步：加载预训练模型自带的分词器和模型 ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
print(f"参数：{sum(p.numel() for p in model.parameters())}")
print("分词示例：", tokenizer.tokenize(reviews[idx_train[0]]))


def batches(idxs, batch_size=32, shuffle=False):
    order = torch.randperm(len(idxs)).tolist() if shuffle else range(len(idxs))
    order = list(order)
    for i in range(0, len(order), batch_size):
        b = [idxs[j] for j in order[i:i + batch_size]]
        enc = tokenizer([reviews[k] for k in b], padding=True, truncation=True, max_length=64,
                        return_tensors="pt").to(device)
        yield enc, torch.tensor([labels[k] for k in b], device=device)


@torch.no_grad()
def evaluate(idxs):
    model.eval()
    correct = 0
    for enc, y in batches(idxs, 128):
        correct += (model(**enc).logits.argmax(1) == y).sum().item()
    return correct / len(idxs)


# ---------- 第 3 步：微调 2 轮，学习率很小 ----------
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
start = time.time()
for epoch in range(1, 3):
    model.train()
    for enc, y in batches(idx_train, shuffle=True):
        loss = model(**enc, labels=y).loss          # 模型自带交叉熵损失
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"第 {epoch} 轮：验证准确率 {evaluate(idx_val):.4f}，已用 {time.time() - start:.0f} 秒")

print(f"测试准确率：{evaluate(idx_test):.4f}")
