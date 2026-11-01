"""由概念表（concepts-draft.csv）和附录 D 术语表生成学习图谱 data.js。

用法：python tools/build_graph.py && python tools/graph_text.py
节点 = 术语表全部术语；先修关系取自概念表（概念名按 ALIAS 对应到术语名，
不在术语表里的概念把它的先修关系传下去），术语表里有、概念表里没有的术语用 EXTRA 补先修。
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import glossary_links as g  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LG = ROOT / "docs/learning-graph"
CATS = [("FOUND", "基础概念"), ("DATA", "数据与特征"), ("REG", "回归"), ("CLS", "分类"), ("ENS", "树模型与集成"),
        ("EVAL", "评估与调优"), ("UNSUP", "无监督学习"), ("NN", "神经网络"), ("SEQ", "序列与时间序列"), ("APP", "强化学习与责任")]
ALIAS = {"回归任务": ["回归"], "分类任务": ["分类"], "聚类任务": ["聚类"], "箱线图": ["箱线图法"], "缺失值填充": ["缺失值"],
         "交叉熵损失": ["交叉熵"], "Softmax 函数": ["Softmax"], "根节点与叶节点": ["根节点", "叶节点"],
         "硬间隔与软间隔": ["硬间隔", "软间隔"], "可解释性": ["特征重要性"], "Stacking": ["集成学习"], "投票法": ["集成学习"]}
EXTRA = {  # 术语: (先修, 类别)
    "模型参数": (["模型"], "FOUND"), "回归": (["监督学习"], "FOUND"), "分类": (["监督学习"], "FOUND"),
    "自监督学习": (["无监督学习"], "FOUND"), "数据透视": (["探索性数据分析"], "DATA"), "数据清洗": (["数据质量"], "DATA"),
    "3σ 法则": (["异常值"], "DATA"), "箱线图法": (["异常值", "描述性统计"], "DATA"),
    "高基数类别特征": (["类别型特征"], "DATA"), "目标编码": (["高基数类别特征", "数据泄露"], "DATA"),
    "特征缩放": (["数值型特征"], "DATA"), "特征工程": (["特征"], "DATA"), "交叉特征": (["特征构造"], "DATA"),
    "残差": (["线性回归"], "REG"), "多重共线性": (["多元线性回归", "相关系数"], "REG"), "残差图": (["残差"], "REG"),
    "正类": (["二分类"], "CLS"), "几率": (["概率输出"], "CLS"), "假阳性率": (["混淆矩阵"], "EVAL"),
    "SMOTE": (["过采样与欠采样"], "EVAL"), "预剪枝": (["剪枝"], "ENS"), "后剪枝": (["剪枝"], "ENS"),
    "弱学习器": (["集成学习"], "ENS"), "合页损失": (["软间隔", "损失函数"], "CLS"), "一对一": (["多分类", "支持向量机"], "CLS"),
    "置换重要性": (["特征重要性"], "EVAL"), "数据漂移": (["模型部署"], "EVAL")}
EXTRA_EDGES = [("特征缩放", "标准化"), ("特征缩放", "归一化"), ("假阳性率", "ROC 曲线"), ("弱学习器", "Boosting"),
               ("残差", "梯度提升树"), ("根节点", "叶节点"), ("残差", "均方误差")]


def main():
    G = g.parse()
    terms = {t: (ch, a, d) for ch, lst in G.items() for t, a, d in lst}
    rows = list(csv.DictReader(open(LG / "concepts-draft.csv", encoding="utf-8")))
    byid = {r["ConceptID"]: r for r in rows}
    code = [c for c, _ in CATS]

    def resolve(cid, seen=()):
        r = byid[cid]
        name = r["ConceptLabel"]
        if name in terms:
            return [name]
        if name in ALIAS:
            return ALIAS[name]
        out = []                                      # 不在术语表里：把它的先修传下去
        for d in filter(None, r["Dependencies"].split("|")):
            if d not in seen:
                out += resolve(d, seen + (cid,))
        return out

    cat, edges = {}, []
    for r in rows:
        for t in resolve(r["ConceptID"]) if (r["ConceptLabel"] in terms or r["ConceptLabel"] in ALIAS) else []:
            cat.setdefault(t, r["TaxonomyID"])
            for d in filter(None, r["Dependencies"].split("|")):
                for p in resolve(d):
                    edges.append((p, t))
    for t, (pre, c) in EXTRA.items():
        cat[t] = c
        edges += [(p, t) for p in pre]
    edges += EXTRA_EDGES
    # 去掉自环、重复、指向更早章节的边（后续概念不能出现在先修概念之前的章）
    seen, E = set(), []
    for a, b in edges:
        if a == b or (a, b) in seen or a not in terms or b not in terms:
            continue
        if terms[a][0] > terms[b][0]:
            continue
        seen.add((a, b)); E.append([a, b])
    nodes = []
    for ch, lst in sorted(G.items()):
        for t, a, d in lst:
            nodes.append({"id": t, "ch": ch, "cat": code.index(cat.get(t, "FOUND")), "a": a, "d": d})
    old = (LG / "data.js").read_text(encoding="utf-8")
    head = old[:old.index("var NODES")]
    head = head.replace(head[head.index("var CATS"):head.index(";", head.index("var CATS")) + 1],
                        "var CATS = " + json.dumps([n for _, n in CATS], ensure_ascii=False) + ";")
    body = "var NODES = [\n" + ",\n".join("  " + json.dumps(n, ensure_ascii=False) for n in nodes) + "\n];\n"
    body += "var EDGES = " + json.dumps(E, ensure_ascii=False) + ";\n"
    (LG / "data.js").write_text(head + body, encoding="utf-8")
    nocat = [t for t in terms if t not in cat]
    print(f"节点 {len(nodes)}，先修关系 {len(E)}；没有分类的术语（按基础概念）：{nocat}")
    orphan = [n["id"] for n in nodes if not any(n["id"] in e for e in E)]
    print("孤立节点：", orphan)


main()
