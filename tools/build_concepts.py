"""根据下面的概念清单生成 docs/learning-graph/learning-graph.csv。

每行：概念, 章, 类别, 先修概念（用 | 分隔，必须是前面已出现的概念）, 归属
归属为空表示本课主讲；"链接:课程名" 表示由其他课主讲、本课只留速记。
"""
import csv
import sys
from pathlib import Path

CATS = {
    "FOUND": "基础概念",
    "DATA": "数据处理与特征工程",
    "REG": "回归",
    "CLS": "分类算法",
    "ENS": "树模型与集成学习",
    "EVAL": "评估与调优",
    "UNSUP": "无监督学习与关联规则",
    "NN": "神经网络",
    "SEQ": "序列与时间序列",
    "APP": "强化学习、部署与伦理",
}

C = """
1|机器学习|FOUND||
1|人工智能|FOUND||链接:人工智能导论
1|规则方法|FOUND|人工智能|
1|数据驱动|FOUND|机器学习|规则方法
1|样本|FOUND|机器学习|
1|特征|FOUND|样本|
1|标签|FOUND|样本|
1|数据集|FOUND|样本|
1|模型|FOUND|机器学习|
1|训练|FOUND|模型|数据集
1|预测|FOUND|模型|
1|泛化能力|FOUND|训练|预测
1|监督学习|FOUND|标签|训练
1|无监督学习|FOUND|机器学习|特征
1|半监督学习|FOUND|监督学习|无监督学习
1|强化学习|FOUND|机器学习|
1|回归任务|FOUND|监督学习|
1|分类任务|FOUND|监督学习|
1|聚类任务|FOUND|无监督学习|
1|机器学习项目流程|FOUND|机器学习|
1|训练集|FOUND|数据集|
1|测试集|FOUND|数据集|
1|scikit-learn|FOUND|机器学习|
1|Jupyter Notebook|FOUND|机器学习项目流程|链接:Python 程序设计
1|NumPy|FOUND||链接:Python 程序设计
1|pandas|FOUND|NumPy|
1|深度学习|FOUND|机器学习|
1|大模型|FOUND|深度学习|链接:自然语言处理
2|数据质量|DATA|数据集|
2|探索性数据分析|DATA|pandas|数据质量
2|描述性统计|DATA|探索性数据分析|
2|缺失值|DATA|数据质量|
2|缺失值填充|DATA|缺失值|描述性统计
2|异常值|DATA|数据质量|描述性统计
2|箱线图|DATA|异常值|
2|重复值|DATA|数据质量|
2|数值型特征|DATA|特征|
2|类别型特征|DATA|特征|
2|独热编码|DATA|类别型特征|
2|标签编码|DATA|类别型特征|
2|标准化|DATA|数值型特征|描述性统计
2|归一化|DATA|数值型特征|
2|分箱|DATA|数值型特征|
2|特征构造|DATA|特征|
2|日期特征|DATA|特征构造|
2|特征选择|DATA|特征|
2|相关系数|DATA|描述性统计|
2|过滤法|DATA|特征选择|相关系数
2|包裹法|DATA|特征选择|
2|嵌入法|DATA|特征选择|
2|数据泄露|DATA|训练集|测试集|特征构造
2|数据清洗规范|DATA|缺失值填充|异常值|重复值
2|数据可视化|DATA|探索性数据分析|
3|线性回归|REG|回归任务|数值型特征
3|一元线性回归|REG|线性回归|
3|多元线性回归|REG|一元线性回归|
3|权重与偏置|REG|线性回归|
3|损失函数|REG|模型|
3|均方误差|REG|损失函数|
3|最小二乘法|REG|均方误差|一元线性回归
3|梯度|REG|损失函数|
3|梯度下降|REG|梯度|
3|学习率|REG|梯度下降|
3|随机梯度下降|REG|梯度下降|
3|小批量梯度下降|REG|随机梯度下降|
3|多项式回归|REG|多元线性回归|特征构造
3|过拟合|REG|泛化能力|多项式回归
3|欠拟合|REG|泛化能力|
3|正则化|REG|过拟合|
3|岭回归|REG|正则化|
3|Lasso 回归|REG|正则化|
3|平均绝对误差|REG|回归任务|
3|均方根误差|REG|均方误差|
3|决定系数|REG|均方误差|
4|逻辑回归|CLS|线性回归|分类任务
4|Sigmoid 函数|CLS|逻辑回归|
4|交叉熵损失|CLS|逻辑回归|损失函数
4|决策边界|CLS|逻辑回归|
4|二分类|CLS|分类任务|
4|多分类|CLS|二分类|
4|Softmax 函数|CLS|多分类|Sigmoid 函数
4|概率输出|CLS|Sigmoid 函数|
4|分类阈值|CLS|概率输出|
4|混淆矩阵|EVAL|二分类|
4|准确率|EVAL|混淆矩阵|
4|精确率|EVAL|混淆矩阵|
4|召回率|EVAL|混淆矩阵|
4|F1 值|EVAL|精确率|召回率
4|ROC 曲线|EVAL|分类阈值|召回率
4|AUC|EVAL|ROC 曲线|
4|PR 曲线|EVAL|精确率|召回率|分类阈值
4|类别不平衡|EVAL|二分类|准确率
4|过采样与欠采样|EVAL|类别不平衡|
4|类别权重|EVAL|类别不平衡|
4|错误案例分析|EVAL|混淆矩阵|
4|模型测试报告|EVAL|错误案例分析|F1 值
5|决策树|ENS|分类任务|特征
5|根节点与叶节点|ENS|决策树|
5|信息熵|ENS|决策树|
5|信息增益|ENS|信息熵|
5|信息增益率|ENS|信息增益|
5|基尼系数|ENS|决策树|
5|ID3 算法|ENS|信息增益|
5|C4.5 算法|ENS|信息增益率|ID3 算法
5|CART 算法|ENS|基尼系数|
5|回归树|ENS|CART 算法|回归任务
5|剪枝|ENS|决策树|过拟合
5|树的深度|ENS|决策树|
5|集成学习|ENS|决策树|
5|Bagging|ENS|集成学习|
5|自助采样|ENS|Bagging|
5|随机森林|ENS|Bagging|
5|袋外误差|ENS|随机森林|自助采样
5|Boosting|ENS|集成学习|
5|AdaBoost|ENS|Boosting|
5|梯度提升树|ENS|Boosting|梯度下降|回归树
5|XGBoost|ENS|梯度提升树|正则化
5|LightGBM|ENS|梯度提升树|
5|Stacking|ENS|集成学习|
5|投票法|ENS|集成学习|
5|特征重要性|ENS|随机森林|
5|可解释性|ENS|特征重要性|
6|支持向量机|CLS|决策边界|
6|间隔|CLS|支持向量机|
6|支持向量|CLS|间隔|
6|硬间隔与软间隔|CLS|间隔|
6|惩罚参数 C|CLS|硬间隔与软间隔|正则化
6|线性可分|CLS|决策边界|
6|核函数|CLS|支持向量机|线性可分
6|核技巧|CLS|核函数|
6|线性核|CLS|核函数|
6|多项式核|CLS|核函数|
6|RBF 核|CLS|核函数|
6|gamma 参数|CLS|RBF 核|
6|支持向量回归|CLS|支持向量机|回归任务
7|K 近邻|CLS|分类任务|特征
7|距离度量|CLS|K 近邻|标准化
7|欧氏距离|CLS|距离度量|
7|曼哈顿距离|CLS|距离度量|
7|余弦相似度|CLS|距离度量|
7|K 值选择|CLS|K 近邻|过拟合|欠拟合
7|维度灾难|CLS|K 近邻|
7|条件概率|CLS||链接:人工智能数学基础
7|贝叶斯定理|CLS|条件概率|
7|先验概率与后验概率|CLS|贝叶斯定理|
7|朴素贝叶斯|CLS|贝叶斯定理|分类任务
7|条件独立假设|CLS|朴素贝叶斯|
7|拉普拉斯平滑|CLS|朴素贝叶斯|
7|高斯朴素贝叶斯|CLS|朴素贝叶斯|数值型特征
7|多项式朴素贝叶斯|CLS|朴素贝叶斯|
7|关联规则|UNSUP|无监督学习|条件概率
7|项集|UNSUP|关联规则|
7|支持度|UNSUP|项集|
7|频繁项集|UNSUP|项集|支持度
7|置信度|UNSUP|支持度|条件概率
7|提升度|UNSUP|置信度|
7|Apriori 算法|UNSUP|频繁项集|
7|FP-Growth 算法|UNSUP|频繁项集|
7|购物篮分析|UNSUP|关联规则|
7|推荐系统|UNSUP|购物篮分析|链接:人工智能数据分析与处理
8|聚类|UNSUP|聚类任务|距离度量
8|K-Means|UNSUP|聚类|欧氏距离
8|簇中心|UNSUP|K-Means|
8|肘部法则|UNSUP|K-Means|
8|轮廓系数|UNSUP|聚类|
8|层次聚类|UNSUP|聚类|
8|树状图|UNSUP|层次聚类|
8|DBSCAN|UNSUP|聚类|
8|密度可达|UNSUP|DBSCAN|
8|噪声点|UNSUP|DBSCAN|异常值
8|客户分群|UNSUP|K-Means|
8|降维|UNSUP|维度灾难|
8|主成分分析|UNSUP|降维|标准化
8|方差解释率|UNSUP|主成分分析|
8|t-SNE|UNSUP|降维|
8|异常检测|UNSUP|聚类|异常值
8|孤立森林|UNSUP|异常检测|随机森林
9|偏差与方差|EVAL|过拟合|欠拟合
9|验证集|EVAL|训练集|测试集
9|交叉验证|EVAL|验证集|
9|K 折交叉验证|EVAL|交叉验证|
9|分层抽样|EVAL|K 折交叉验证|类别不平衡
9|时间序列交叉验证|EVAL|K 折交叉验证|
9|学习曲线|EVAL|偏差与方差|
9|验证曲线|EVAL|学习曲线|
9|超参数|EVAL|模型|
9|网格搜索|EVAL|超参数|交叉验证
9|随机搜索|EVAL|网格搜索|
9|Optuna 调参|EVAL|随机搜索|
9|流水线|EVAL|数据泄露|标准化
9|列转换器|EVAL|流水线|独热编码
9|模型保存|APP|模型|
9|joblib|APP|模型保存|
9|模型接口|APP|模型保存|
9|FastAPI|APP|模型接口|
9|Gradio|APP|模型接口|
9|模型部署|APP|模型接口|链接:人工智能开发框架技术
9|SHAP 值|EVAL|可解释性|
10|神经网络|NN|逻辑回归|
10|神经元|NN|神经网络|
10|感知机|NN|神经元|决策边界
10|异或问题|NN|感知机|线性可分
10|多层感知机|NN|感知机|异或问题
10|激活函数|NN|神经元|Sigmoid 函数
10|ReLU|NN|激活函数|
10|隐藏层|NN|多层感知机|
10|前向传播|NN|多层感知机|
10|反向传播|NN|前向传播|梯度下降
10|链式法则|NN|反向传播|
10|梯度消失|NN|反向传播|激活函数
10|优化器|NN|小批量梯度下降|
10|Adam|NN|优化器|
10|轮次与批大小|NN|小批量梯度下降|
10|Dropout|NN|过拟合|隐藏层
10|早停|NN|验证集|过拟合
10|批归一化|NN|标准化|隐藏层
10|PyTorch|NN|NumPy|
10|张量|NN|PyTorch|
10|自动求导|NN|张量|反向传播
10|Keras|NN|神经网络|
10|TensorFlow|NN|Keras|
11|卷积神经网络|NN|多层感知机|
11|图像的数字表示|NN|张量|
11|卷积|NN|卷积神经网络|图像的数字表示
11|卷积核|NN|卷积|
11|特征图|NN|卷积核|
11|步长与填充|NN|卷积|
11|池化|NN|特征图|
11|全连接层|NN|多层感知机|
11|LeNet|NN|卷积|池化|全连接层
11|数据增强|NN|卷积神经网络|过拟合
11|迁移学习|NN|卷积神经网络|
11|预训练模型|NN|迁移学习|
11|循环神经网络|SEQ|多层感知机|
11|序列数据|SEQ|循环神经网络|
11|隐藏状态|SEQ|循环神经网络|
11|时间步|SEQ|序列数据|
11|长期依赖|SEQ|隐藏状态|梯度消失
11|LSTM|SEQ|长期依赖|
11|门控机制|SEQ|LSTM|
11|GRU|SEQ|门控机制|
11|词嵌入|SEQ|序列数据|链接:自然语言处理
12|时间序列|SEQ|序列数据|
12|趋势|SEQ|时间序列|
12|季节性|SEQ|时间序列|
12|平稳性|SEQ|趋势|季节性
12|差分|SEQ|平稳性|
12|滞后特征|SEQ|时间序列|特征构造
12|滑动窗口|SEQ|滞后特征|
12|移动平均|SEQ|时间序列|
12|指数平滑|SEQ|移动平均|
12|ARIMA|SEQ|差分|移动平均
12|自相关|SEQ|时间序列|相关系数
12|基于特征的时序预测|SEQ|滑动窗口|LightGBM
12|LSTM 时序预测|SEQ|LSTM|滑动窗口
12|时序评价指标|SEQ|平均绝对误差|时间序列交叉验证
12|智能体|APP|强化学习|
12|环境|APP|智能体|
12|状态|APP|环境|
12|动作|APP|智能体|
12|奖励|APP|动作|状态
12|策略|APP|状态|动作
12|探索与利用|APP|策略|
12|马尔可夫决策过程|APP|状态|动作|奖励
12|Q 学习|APP|马尔可夫决策过程|探索与利用
12|Q 表|APP|Q 学习|
12|深度强化学习|APP|Q 学习|神经网络
12|人类反馈强化学习|APP|深度强化学习|大模型
12|算法公平性|APP|可解释性|
12|数据合规|APP|数据清洗规范|
"""

def main(out):
    rows = []
    idx = {}
    for line in C.strip().splitlines():
        parts = line.split("|")
        ch, label, cat = int(parts[0]), parts[1], parts[2]
        rest = [p for p in parts[3:] if p]
        owner = ""
        if rest and rest[-1].startswith("链接:"):
            owner = rest.pop()[3:]
        if label in idx:
            sys.exit(f"重复概念：{label}")
        for d in rest:
            if d not in idx:
                sys.exit(f"{label} 的先修 {d} 未在前面出现")
        idx[label] = len(rows) + 1
        rows.append((idx[label], label, "|".join(str(idx[d]) for d in rest), cat, ch, owner))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ConceptID", "ConceptLabel", "Dependencies", "TaxonomyID", "Chapter", "Owner"])
        w.writerows(rows)
    by_ch = {}
    for r in rows:
        by_ch[r[4]] = by_ch.get(r[4], 0) + 1
    edges = sum(len(r[2].split("|")) for r in rows if r[2])
    print(f"{len(rows)} 个概念，{edges} 条先修；各章：{by_ch}")
    print("链接到其他课：", [(r[1], r[5]) for r in rows if r[5]])

if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "docs/learning-graph/concepts-draft.csv"))
