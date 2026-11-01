---
title: 附录 C 术语表
description: 本书主要术语的中英文名称、简明定义和例子，按章节顺序排列
---

# 附录 C 术语表

本表按章节顺序列出本书的主要术语，每个术语给出英文名称、简明定义和例子。正文中带虚线下划线的术语，把鼠标停在上面就能看到定义，点击可以跳到本表。

通用术语（如“过拟合”“精确率”）的译名和定义以本系列教材共用的《系列公共术语表》为准，各书保持一致。

## 第 1 章 初识神经网络：从一个神经元说起

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-神经网络"></span>神经网络 | Neural Network | 由许多“神经元”一层层连接而成的计算模型，通过调整连接的权重从数据中学习 | 第 1 章认手写数字的 784-128-64-10 网络 |
| <span id="g-深度学习"></span>深度学习 | Deep Learning | 使用多层神经网络从数据中自动学习特征的机器学习方法 | 图像识别、语音识别和大模型都基于深度学习 |
| <span id="g-神经元"></span>神经元 | Neuron | 神经网络的基本单元：把输入乘以权重相加，加上偏置，再经过激活函数输出一个数 | 输入 (2, 3)、权重 (0.5, −1)、偏置 1，加权和为 −1 |
| <span id="g-权重"></span>权重 | Weight | 神经元给每个输入乘的系数，表示这个输入有多重要，是训练要调整的参数 | 权重为负表示这个输入越大，输出越小 |
| <span id="g-偏置"></span>偏置 | Bias | 神经元加权求和后再加上的一个数，用来调整神经元“多容易被激活” | 相当于直线 y = wx + b 中的 b |
| <span id="g-激活函数"></span>激活函数 | Activation Function | 神经元在加权求和之后进行的非线性变换，让网络能表示弯曲的关系 | ReLU、Sigmoid、Tanh |
| <span id="g-ReLU"></span>ReLU | Rectified Linear Unit | 最常用的激活函数：输入小于 0 输出 0，否则原样输出 | ReLU(−2) = 0，ReLU(3) = 3 |
| <span id="g-Sigmoid"></span>Sigmoid | Sigmoid | 把任意数压缩到 0～1 之间的 S 形激活函数，常用来表示概率 | Sigmoid(0) = 0.5 |
| <span id="g-参数"></span>参数 | Parameter | 网络中通过训练学出来的数，包括所有权重和偏置 | 第 1 章的网络有 109 386 个参数 |
| <span id="g-输入层"></span>输入层 | Input Layer | 接收原始数据的一层，神经元个数等于每个样本的特征数 | 手写数字 28×28 = 784 个像素 |
| <span id="g-隐藏层"></span>隐藏层 | Hidden Layer | 输入层和输出层之间的层，负责逐步提取特征 | 784-128-64-10 网络有两个隐藏层 |
| <span id="g-输出层"></span>输出层 | Output Layer | 网络的最后一层，给出最终结果 | 10 个神经元对应数字 0～9 的得分 |
| <span id="g-前向传播"></span>前向传播 | Forward Propagation | 数据从输入层开始，一层层计算到输出层的过程 | 输入一张图片，算出 10 个数字的概率 |

## 第 2 章 训练的本质：损失函数与梯度下降

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-损失函数"></span>损失函数 | Loss Function | 衡量模型预测与正确答案差距的函数，值越小说明预测越准 | 均方误差、交叉熵 |
| <span id="g-均方误差"></span>均方误差 | Mean Squared Error, MSE | 预测值与真实值之差的平方的平均值，常用于预测数值的任务 | 预测 18、真实 20，平方误差为 4 |
| <span id="g-交叉熵"></span>交叉熵 | Cross-Entropy | 分类任务常用的损失函数，正确类别的概率越大，损失越小 | 正确类别概率 0.9 时损失约 0.105 |
| <span id="g-Softmax"></span>Softmax | Softmax | 把一组得分变成加起来等于 1 的概率的函数 | 得分 (2, 1, 0) 变成约 (0.67, 0.24, 0.09) |
| <span id="g-梯度"></span>梯度 | Gradient | 损失对每个参数的变化率，指出参数往哪个方向改、损失增加得最快 | 梯度为正时应把参数调小 |
| <span id="g-梯度下降"></span>梯度下降 | Gradient Descent | 沿着梯度的反方向一小步一小步地更新参数，让损失逐渐变小的方法 | 新参数 = 旧参数 − 学习率 × 梯度 |
| <span id="g-学习率"></span>学习率 | Learning Rate | 每次更新参数时迈出的步子大小，是最重要的超参数之一 | 学习率太大会发散，太小会学得很慢 |
| <span id="g-超参数"></span>超参数 | Hyperparameter | 训练前由人设定、不由训练学出来的设置 | 学习率、批大小、轮次、隐藏层大小 |
| <span id="g-批大小"></span>批大小 | Batch Size | 每更新一次参数所用的样本数 | 批大小 64：每看 64 张图片更新一次 |
| <span id="g-轮次"></span>轮次 | Epoch | 把全部训练数据完整地用一遍，叫一轮 | 6 万张图片、批大小 64，一轮约更新 938 次 |
| <span id="g-张量"></span>张量 | Tensor | PyTorch 中存放数据的多维数组，可以在 GPU 上计算并自动求导 | 一批图片是形状 64×1×28×28 的张量 |
| <span id="g-优化器"></span>优化器 | Optimizer | 按照梯度更新参数的工具，封装了不同的更新方法 | torch.optim.SGD、torch.optim.Adam |
| <span id="g-准确率"></span>准确率 | Accuracy | 预测正确的样本占全部样本的比例 | 1 万张测试图认对 9709 张，准确率 97.09% |

## 第 3 章 反向传播：误差怎么往回传

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-反向传播"></span>反向传播 | Backpropagation | 从输出层开始，按链式法则把损失的梯度一层层往回传、算出每个参数梯度的方法 | PyTorch 中 loss.backward() 完成反向传播 |
| <span id="g-计算图"></span>计算图 | Computational Graph | 把计算过程画成由运算节点和箭头组成的流程图，是自动求导的基础 | y = (x + 1)² 的计算图有“加 1”“平方”两个节点 |
| <span id="g-链式法则"></span>链式法则 | Chain Rule | 复合计算的变化率等于各步变化率相乘 | 像齿轮传动：总传动比等于各级传动比相乘 |
| <span id="g-自动求导"></span>自动求导 | Automatic Differentiation | 框架记录计算图后自动按链式法则计算梯度的功能 | 设置 requires_grad=True 后调用 backward() |
| <span id="g-数值差分"></span>数值差分 | Numerical Differentiation | 把参数改动一点点，看损失变化多少来估计梯度的方法，常用来核对梯度 | 实验 3-1 中与自动求导的差别只有 10⁻¹¹ 量级 |
| <span id="g-梯度消失"></span>梯度消失 | Vanishing Gradient | 梯度往回传时一层层变小，前面的层几乎学不动的现象 | 10 层 Sigmoid 网络第 1 层梯度只有第 10 层的十亿分之四 |
| <span id="g-梯度爆炸"></span>梯度爆炸 | Exploding Gradient | 梯度往回传时一层层变大，导致参数剧烈跳动、损失变成 nan 的现象 | RNN 处理长序列时容易出现 |
| <span id="g-批归一化"></span>批归一化 | Batch Normalization | 把每层的输出按一批数据调整为平均值 0、标准差 1 左右，再往下传 | nn.BatchNorm1d、nn.BatchNorm2d |
| <span id="g-残差连接"></span>残差连接 | Residual Connection | 让输入“抄近路”直接加到几层之后的输出上，使梯度能直接传回前面 | ResNet 和 Transformer 的每一层都有残差连接 |
| <span id="g-梯度裁剪"></span>梯度裁剪 | Gradient Clipping | 梯度太大时把它按比例缩小到规定范围内，防止梯度爆炸 | clip_grad_norm_(params, 1.0) |
| <span id="g-动量"></span>动量 | Momentum | 更新参数时保留一部分上一次的方向，像下坡时的惯性，减少来回摆动 | SGD(momentum=0.9) |
| <span id="g-Adam"></span>Adam | Adam | 结合动量和自适应学习率的优化器，是最常用的默认选择 | torch.optim.Adam(lr=0.001) |

## 第 4 章 把模型训练好：过拟合与调优

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-泛化"></span>泛化 | Generalization | 模型在训练时没见过的新数据上也表现好的能力 | 训练集 100% 但测试集 60%，泛化能力差 |
| <span id="g-训练集"></span>训练集 | Training Set | 用来计算损失和梯度、更新参数的数据 | 平时的练习题 |
| <span id="g-验证集"></span>验证集 | Validation Set | 训练中用来比较超参数、决定何时停止的数据 | 模拟考试 |
| <span id="g-测试集"></span>测试集 | Test Set | 一切选择做完后只用一次、报告最终成绩的数据 | 正式考试 |
| <span id="g-欠拟合"></span>欠拟合 | Underfitting | 模型连训练集都没学好，训练和验证误差都高 | 用一条直线去拟合弯曲的数据 |
| <span id="g-过拟合"></span>过拟合 | Overfitting | 模型把训练集“背”下来，在新数据上表现变差 | 训练损失一直降，验证损失先降后升 |
| <span id="g-正则化"></span>正则化 | Regularization | 减轻过拟合的各种方法的统称 | Dropout、权重衰减、数据增强、早停 |
| <span id="g-Dropout"></span>Dropout | Dropout | 训练时每一批随机让一部分神经元输出为 0 的正则化方法，也叫随机失活 | nn.Dropout(0.5)，评估前要 model.eval() |
| <span id="g-权重衰减"></span>权重衰减 | Weight Decay | 每次更新时把权重往 0 的方向拉一点，让权重保持较小 | Adam(..., weight_decay=1e-3) |
| <span id="g-数据增强"></span>数据增强 | Data Augmentation | 对训练样本做不改变答案的随机变换，“变”出更多训练数据 | 随机旋转 ±10 度、平移、水平翻转 |
| <span id="g-早停"></span>早停 | Early Stopping | 验证指标连续若干轮不再变好就停止训练，并恢复最好那一轮的参数 | 耐心 5 轮：连续 5 轮没创新低就停 |
| <span id="g-网格搜索"></span>网格搜索 | Grid Search | 把几个超参数的候选值两两组合，逐一训练比较的调参方法 | 3 个学习率 × 2 种大小 × 2 种 Dropout = 12 组 |

## 第 5 章 卷积神经网络：为图像而生

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-通道"></span>通道 | Channel | 图像或特征图的“层数”，彩色图片有红、绿、蓝 3 个通道 | 64 张 CIFAR-10 图片的形状 64×3×32×32 |
| <span id="g-卷积"></span>卷积 | Convolution | 用一个小窗口在图上滑动，每个位置对应相乘再相加，得到一张新图的运算 | 3×3 卷积核在 5×5 图上滑动，得到 3×3 的结果 |
| <span id="g-卷积核"></span>卷积核 | Kernel | 卷积中滑动的小数字方阵，也叫滤波器，数值通过训练学出来 | 竖直边缘卷积核每行是 (−1, 0, 1) |
| <span id="g-特征图"></span>特征图 | Feature Map | 卷积层或池化层输出的“图”，表示某种图案在各位置的强弱 | 16 个卷积核输出 16 张特征图 |
| <span id="g-局部连接"></span>局部连接 | Local Connectivity | 每个输出只和输入中的一小块区域相连 | 3×3 卷积只看 9 个像素 |
| <span id="g-权值共享"></span>权值共享 | Weight Sharing | 特征图上所有位置使用同一个卷积核的参数 | 一个 3×3 卷积核只有 10 个参数，与图片大小无关 |
| <span id="g-填充"></span>填充 | Padding | 卷积前在图片四周补几圈 0，常用来保持尺寸不变 | 3×3 卷积配填充 1，32×32 仍为 32×32 |
| <span id="g-步幅"></span>步幅 | Stride | 卷积核每次移动的格数 | 步幅 2 让特征图边长减半 |
| <span id="g-池化"></span>池化 | Pooling | 把特征图分成小块、每块取一个代表值来缩小尺寸的操作，没有参数 | 2×2 最大池化把 32×32 变为 16×16 |
| <span id="g-最大池化"></span>最大池化 | Max Pooling | 每个小块取最大值的池化 | {1, 3, 4, 2} 取 4 |
| <span id="g-卷积神经网络"></span>卷积神经网络 | Convolutional Neural Network, CNN | 由卷积层、池化层和全连接层组成，专门处理图像等网格数据的网络 | LeNet、AlexNet、VGG、ResNet |
| <span id="g-ResNet"></span>ResNet | Residual Network | 2015 年提出的带残差连接的深层卷积网络，可达上百层 | 第 7 章迁移学习用的 ResNet18 |

## 第 6 章 序列模型：从 RNN 到 Transformer

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-序列数据"></span>序列数据 | Sequential Data | 按顺序排列、顺序本身带有信息的数据 | 一句话、一段语音、每天的气温 |
| <span id="g-循环神经网络"></span>循环神经网络 | Recurrent Neural Network, RNN | 一步步读序列、每一步把当前输入和旧的隐藏状态结合成新隐藏状态的网络 | nn.RNN |
| <span id="g-隐藏状态"></span>隐藏状态 | Hidden State | RNN 读到某一步时的“笔记”，概括了到目前为止读过的内容 | 实验 6-1 中是 32 维向量 |
| <span id="g-LSTM"></span>LSTM | Long Short-Term Memory | 长短期记忆网络，用记忆细胞和三道门管理信息，能记住更长距离的内容 | nn.LSTM |
| <span id="g-记忆细胞"></span>记忆细胞 | Cell State | LSTM 中贯穿各步的“传送带”，信息可以几乎不变地往后传 | 遗忘门接近 1 时记忆几乎原样保留 |
| <span id="g-遗忘门"></span>遗忘门 | Forget Gate | LSTM 中决定旧记忆保留多少的门 | 读到新主语时忘掉旧主语 |
| <span id="g-输入门"></span>输入门 | Input Gate | LSTM 中决定写入多少新内容的门 | 读到“成都”时记下来 |
| <span id="g-输出门"></span>输出门 | Output Gate | LSTM 中决定这一步拿出多少记忆来用的门 | 需要填空时取出“成都” |
| <span id="g-注意力"></span>注意力 | Attention | 回头看所有位置，按相关程度打分、归一化后加权平均的计算方法 | 翻译时每个词去看原文中最相关的词 |
| <span id="g-自注意力"></span>自注意力 | Self-Attention | 一句话中的每个词都去看同一句话里的所有词的注意力 | “它”去找它指代的“小猫” |
| <span id="g-多头注意力"></span>多头注意力 | Multi-Head Attention | 同时做多组注意力、结果拼起来，不同的头关注不同的关系 | nn.MultiheadAttention(64, 4) |
| <span id="g-位置编码"></span>位置编码 | Positional Encoding | 给每个位置加上的向量，告诉网络每个字在第几位 | 实验 6-3 中可学习的位置向量 |
| <span id="g-Transformer"></span>Transformer | Transformer | 完全基于注意力的序列模型，由多头自注意力、前馈网络、残差连接与归一化重复堆叠而成 | BERT 用编码器，GPT 用解码器 |

## 第 7 章 图像分类实战与迁移学习

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-图像分类"></span>图像分类 | Image Classification | 判断一张图片属于哪一类的任务 | 水果分拣、垃圾分类 |
| <span id="g-目标检测"></span>目标检测 | Object Detection | 找出图中有哪些物体并用方框标出位置的任务 | 安全帽检测 |
| <span id="g-图像分割"></span>图像分割 | Image Segmentation | 判断每个像素属于哪一类的任务 | 医学影像勾画病灶 |
| <span id="g-迁移学习"></span>迁移学习 | Transfer Learning | 把在大数据集上训练好的模型借来，用于新任务的方法 | 用 ImageNet 预训练的 ResNet18 分 10 类照片 |
| <span id="g-预训练模型"></span>预训练模型 | Pre-trained Model | 已经在大规模数据上训练好、可以拿来迁移或微调的模型 | timm 中的 ResNet18 权重 |
| <span id="g-ImageNet"></span>ImageNet | ImageNet | 约 128 万张照片、1000 类的大型图像数据集，常用于预训练 | Imagenette 是从中挑出的 10 类 |
| <span id="g-冻结"></span>冻结 | Freezing | 训练时让预训练部分的参数保持不变，只训练新加的层 | 把参数的 requires_grad 设为 False |
| <span id="g-微调"></span>微调 | Fine-tuning | 在预训练模型基础上，用小学习率继续训练部分或全部参数 | 实验 7-2 解冻全部、学习率 0.0001 |
| <span id="g-混淆矩阵"></span>混淆矩阵 | Confusion Matrix | 按“真实类别 × 预测类别”统计样本数的表，对角线是认对的 | 实验 7-3 的 10×10 矩阵 |
| <span id="g-精确率"></span>精确率 | Precision | 模型判为某类的样本中，真正属于该类的比例（看列） | 链锯的精确率 0.781 |
| <span id="g-召回率"></span>召回率 | Recall | 真正属于某类的样本中，被模型找出来的比例（看行） | 教堂的召回率 0.966 |
| <span id="g-F1-值"></span>F1 值 | F1 Score | 精确率与召回率的调和平均数，兼顾两者 | 精确率、召回率都是 0.8 时 F1 也是 0.8 |

## 第 8 章 文本分类实战与预训练模型

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-分词"></span>分词 | Word Segmentation | 把句子切成一个个词的处理，中文需要专门的分词工具 | jieba.lcut("很快，好吃") |
| <span id="g-词表"></span>词表 | Vocabulary | 训练集中出现的词及其编号的对照表，只能用训练集来建 | 0 号补齐，1 号未知 |
| <span id="g-补齐"></span>补齐 | Padding | 把短序列在后面补 0，使一批数据长度相同 | 5 个词补到 40 个 |
| <span id="g-独热编码"></span>独热编码 | One-hot Encoding | 用只有一个位置是 1、其余全是 0 的长向量表示一个词 | 4000 个词的词表，每个词是 4000 维 |
| <span id="g-词嵌入"></span>词嵌入 | Word Embedding | 给每个词分配一个短而稠密的向量的方法，PyTorch 中是 nn.Embedding | nn.Embedding(4334, 64) |
| <span id="g-词向量"></span>词向量 | Word Vector | 词嵌入得到的向量，意思或用法相近的词向量也相近 | “好吃”与“正宗”“很香”相近 |
| <span id="g-Word2Vec"></span>Word2Vec | Word2Vec | 根据一个词周围经常出现哪些词来学习词向量的方法 | gensim 的 Word2Vec |
| <span id="g-余弦相似度"></span>余弦相似度 | Cosine Similarity | 两个向量点积除以长度之积，只看方向不看长短，范围 −1～1 | (3, 4) 与 (4, 3) 的余弦相似度 0.96 |
| <span id="g-情感分类"></span>情感分类 | Sentiment Classification | 判断一段文字表达的是正面还是负面态度的任务 | 外卖评论分好评、差评 |
| <span id="g-预训练语言模型"></span>预训练语言模型 | Pre-trained Language Model | 先用海量文本预训练、再针对具体任务微调的语言模型 | BERT、RoBERTa |
| <span id="g-BERT"></span>BERT | Bidirectional Encoder Representations from Transformers | 2018 年提出的 Transformer 编码器预训练模型，通过“完形填空”学习语言 | 哈工大的中文 BERT-wwm |

## 第 9 章 生成模型：让网络学会“创作”

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-判别模型"></span>判别模型 | Discriminative Model | 给输入判断类别或数值的模型 | 第 1～8 章的分类和预测模型 |
| <span id="g-生成模型"></span>生成模型 | Generative Model | 生成新的、和训练数据相似的数据的模型 | 自编码器、GAN、扩散模型、大语言模型 |
| <span id="g-自编码器"></span>自编码器 | Autoencoder | 由编码器压缩、解码器还原组成，目标是让输出等于输入的网络 | 784 → 32 → 784 |
| <span id="g-编码器"></span>编码器 | Encoder | 把输入压缩成较短表示的部分 | 自编码器的前半部分；BERT 是 Transformer 编码器 |
| <span id="g-解码器"></span>解码器 | Decoder | 把较短的表示还原或生成输出的部分 | 自编码器的后半部分；GPT 是 Transformer 解码器 |
| <span id="g-潜在空间"></span>潜在空间 | Latent Space | 编码所在的空间，相近的编码对应相似的数据 | 32 维的编码空间 |
| <span id="g-异常检测"></span>异常检测 | Anomaly Detection | 找出与正常数据明显不同的样本，可用还原误差来判断 | 服装图片的还原误差是手写数字的 11 倍多 |
| <span id="g-生成对抗网络"></span>生成对抗网络 | Generative Adversarial Network, GAN | 生成器和判别器互相比赛、共同进步的生成模型 | 实验 9-2 生成手写数字 |
| <span id="g-生成器"></span>生成器 | Generator | GAN 中输入随机数、输出假数据的网络（造假者） | 100 个随机数 → 784 个像素 |
| <span id="g-判别器"></span>判别器 | Discriminator | GAN 中判断输入是真数据还是假数据的网络（鉴定师） | 输出“是真图”的概率 |
| <span id="g-模式崩溃"></span>模式崩溃 | Mode Collapse | GAN 的生成器只会生成少数几种样子的数据 | 不管输入什么都画“1” |
| <span id="g-扩散模型"></span>扩散模型 | Diffusion Model | 正向逐步加噪声、训练网络猜噪声、再从噪声逐步去噪生成数据的模型 | Stable Diffusion |
| <span id="g-深度伪造"></span>深度伪造 | Deepfake | 用生成模型伪造人物的照片、视频或声音 | AI 换脸诈骗 |

## 第 10 章 深度强化学习、伦理与综合项目

| 术语 | 英文 | 定义 | 例子 |
|---|---|---|---|
| <span id="g-强化学习"></span>强化学习 | Reinforcement Learning | 智能体在环境中试错、根据奖励学习如何行动的方法 | AlphaGo 自我对弈 |
| <span id="g-智能体"></span>智能体 | Agent | 强化学习中做决定、采取行动的一方 | 控制小车的程序 |
| <span id="g-环境"></span>环境 | Environment | 智能体所处的世界，对动作做出反应并给出奖励 | CartPole-v1 |
| <span id="g-状态"></span>状态 | State | 环境当前的情况 | 小车位置、速度，杆子角度、角速度 |
| <span id="g-动作"></span>动作 | Action | 智能体可以做的选择 | 向左推或向右推 |
| <span id="g-奖励"></span>奖励 | Reward | 每做一个动作后得到的分数 | 杆子没倒，每步得 1 分 |
| <span id="g-策略"></span>策略 | Policy | 在什么状态下做什么动作的规则 | 杆子往右倒就往右推 |
| <span id="g-Q-值"></span>Q 值 | Q Value | 在某状态下做某动作，预计从现在到最后总共能得的分数 | 目标值 = 奖励 + γ × 下一状态最大 Q 值 |
| <span id="g-DQN"></span>DQN | Deep Q-Network | 用神经网络代替 Q 表来估计 Q 值的深度强化学习方法 | 实验 10-1 的 4 → 64 → 64 → 2 网络 |
| <span id="g-ε-贪心"></span>ε-贪心 | ε-Greedy | 以 ε 的概率随机选动作、否则选 Q 值最大的动作的策略 | ε 从 1 降到 0.05 |
| <span id="g-经验回放"></span>经验回放 | Experience Replay | 把玩过的每一步存起来、训练时随机抽取一批的方法 | 回放池容量 20 000 步 |
| <span id="g-目标网络"></span>目标网络 | Target Network | 专门用来计算学习目标、隔一段时间才同步一次的网络副本 | 每 500 步同步一次 |
| <span id="g-偏见"></span>偏见 | Bias in AI | 模型因训练数据等原因对某些人群做出不公平判断的问题 | 招聘模型给女性打分偏低 |
| <span id="g-基准方法"></span>基准方法 | Baseline | 用来对比的最简单做法，判断模型是否真的有用 | “明天和今天一样”“全部猜最多的类” |
