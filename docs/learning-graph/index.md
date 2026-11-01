---
title: 学习图谱
description: 全书核心概念及其先修关系，点一个概念就能看到学习路径
---

# 学习图谱

学习图谱把全书的核心概念和它们之间的“先修关系”画在一张图上：要理解一个概念，得先懂哪些更基础的概念；顺着箭头往回找，就是一条学习路径。

<iframe class="sim-frame" src="main.html" height="900" loading="lazy" title="学习图谱"></iframe>

[全屏打开](main.html){ target=_blank }

## 怎样使用

- **预习**：学新的一章前，点开本章的概念，看看橙色的先修概念自己是否都掌握了，没掌握的先回去复习；
- **复习**：期末复习时，从最后几章的概念出发，顺着学习路径往回梳理，就能把全书串起来；
- **查漏补缺**：做练习时遇到不懂的概念，在搜索框里输入它，看它依赖哪些更基础的概念。

<!-- graph:start -->

图谱共有 **125 个概念、166 条先修关系**，概念按章排列，颜色表示 7 个类别（网络结构、训练方法、评估与调优、卷积与视觉、序列与语言、生成模型、强化学习与责任）。概念的定义都来自[附录 D 术语表](../appendix/glossary.md)，先修关系由编者根据各章内容整理。

## 文字版

下面按章列出每个概念的直接先修概念，便于打印或用读屏软件阅读。

??? note "第 1 章 初识神经网络（13 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [神经网络](../appendix/glossary.md#g-神经网络) | 网络结构 | — |
    | [深度学习](../appendix/glossary.md#g-深度学习) | 网络结构 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [神经元](../appendix/glossary.md#g-神经元) | 网络结构 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [权重](../appendix/glossary.md#g-权重) | 网络结构 | [神经元](../appendix/glossary.md#g-神经元) |
    | [偏置](../appendix/glossary.md#g-偏置) | 网络结构 | [神经元](../appendix/glossary.md#g-神经元) |
    | [激活函数](../appendix/glossary.md#g-激活函数) | 网络结构 | [神经元](../appendix/glossary.md#g-神经元) |
    | [ReLU](../appendix/glossary.md#g-ReLU) | 网络结构 | [激活函数](../appendix/glossary.md#g-激活函数) |
    | [Sigmoid](../appendix/glossary.md#g-Sigmoid) | 网络结构 | [激活函数](../appendix/glossary.md#g-激活函数) |
    | [参数](../appendix/glossary.md#g-参数) | 网络结构 | [权重](../appendix/glossary.md#g-权重)、[偏置](../appendix/glossary.md#g-偏置) |
    | [输入层](../appendix/glossary.md#g-输入层) | 网络结构 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [隐藏层](../appendix/glossary.md#g-隐藏层) | 网络结构 | [输入层](../appendix/glossary.md#g-输入层) |
    | [输出层](../appendix/glossary.md#g-输出层) | 网络结构 | [隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [前向传播](../appendix/glossary.md#g-前向传播) | 网络结构 | [输出层](../appendix/glossary.md#g-输出层)、[激活函数](../appendix/glossary.md#g-激活函数) |

??? note "第 2 章 损失与梯度下降（13 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [损失函数](../appendix/glossary.md#g-损失函数) | 训练方法 | [前向传播](../appendix/glossary.md#g-前向传播) |
    | [均方误差](../appendix/glossary.md#g-均方误差) | 训练方法 | [损失函数](../appendix/glossary.md#g-损失函数) |
    | [交叉熵](../appendix/glossary.md#g-交叉熵) | 训练方法 | [损失函数](../appendix/glossary.md#g-损失函数)、[Softmax](../appendix/glossary.md#g-Softmax) |
    | [Softmax](../appendix/glossary.md#g-Softmax) | 训练方法 | [输出层](../appendix/glossary.md#g-输出层) |
    | [梯度](../appendix/glossary.md#g-梯度) | 训练方法 | [损失函数](../appendix/glossary.md#g-损失函数)、[参数](../appendix/glossary.md#g-参数) |
    | [梯度下降](../appendix/glossary.md#g-梯度下降) | 训练方法 | [梯度](../appendix/glossary.md#g-梯度) |
    | [学习率](../appendix/glossary.md#g-学习率) | 训练方法 | [梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [超参数](../appendix/glossary.md#g-超参数) | 训练方法 | [学习率](../appendix/glossary.md#g-学习率) |
    | [批大小](../appendix/glossary.md#g-批大小) | 训练方法 | [梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [轮次](../appendix/glossary.md#g-轮次) | 训练方法 | [批大小](../appendix/glossary.md#g-批大小) |
    | [张量](../appendix/glossary.md#g-张量) | 网络结构 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [优化器](../appendix/glossary.md#g-优化器) | 训练方法 | [梯度下降](../appendix/glossary.md#g-梯度下降) |
    | [准确率](../appendix/glossary.md#g-准确率) | 评估与调优 | [输出层](../appendix/glossary.md#g-输出层) |

??? note "第 3 章 反向传播（12 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [反向传播](../appendix/glossary.md#g-反向传播) | 训练方法 | [梯度](../appendix/glossary.md#g-梯度)、[计算图](../appendix/glossary.md#g-计算图)、[链式法则](../appendix/glossary.md#g-链式法则) |
    | [计算图](../appendix/glossary.md#g-计算图) | 训练方法 | [前向传播](../appendix/glossary.md#g-前向传播) |
    | [链式法则](../appendix/glossary.md#g-链式法则) | 训练方法 | [梯度](../appendix/glossary.md#g-梯度) |
    | [自动求导](../appendix/glossary.md#g-自动求导) | 训练方法 | [反向传播](../appendix/glossary.md#g-反向传播)、[张量](../appendix/glossary.md#g-张量) |
    | [数值差分](../appendix/glossary.md#g-数值差分) | 训练方法 | [梯度](../appendix/glossary.md#g-梯度) |
    | [梯度消失](../appendix/glossary.md#g-梯度消失) | 训练方法 | [反向传播](../appendix/glossary.md#g-反向传播)、[Sigmoid](../appendix/glossary.md#g-Sigmoid) |
    | [梯度爆炸](../appendix/glossary.md#g-梯度爆炸) | 训练方法 | [反向传播](../appendix/glossary.md#g-反向传播) |
    | [批归一化](../appendix/glossary.md#g-批归一化) | 网络结构 | [梯度消失](../appendix/glossary.md#g-梯度消失) |
    | [残差连接](../appendix/glossary.md#g-残差连接) | 网络结构 | [梯度消失](../appendix/glossary.md#g-梯度消失) |
    | [梯度裁剪](../appendix/glossary.md#g-梯度裁剪) | 训练方法 | [梯度爆炸](../appendix/glossary.md#g-梯度爆炸) |
    | [动量](../appendix/glossary.md#g-动量) | 训练方法 | [优化器](../appendix/glossary.md#g-优化器) |
    | [Adam](../appendix/glossary.md#g-Adam) | 训练方法 | [动量](../appendix/glossary.md#g-动量) |

??? note "第 4 章 过拟合与调优（12 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [泛化](../appendix/glossary.md#g-泛化) | 评估与调优 | [测试集](../appendix/glossary.md#g-测试集) |
    | [训练集](../appendix/glossary.md#g-训练集) | 评估与调优 | [准确率](../appendix/glossary.md#g-准确率) |
    | [验证集](../appendix/glossary.md#g-验证集) | 评估与调优 | [训练集](../appendix/glossary.md#g-训练集) |
    | [测试集](../appendix/glossary.md#g-测试集) | 评估与调优 | [训练集](../appendix/glossary.md#g-训练集) |
    | [欠拟合](../appendix/glossary.md#g-欠拟合) | 评估与调优 | [验证集](../appendix/glossary.md#g-验证集) |
    | [过拟合](../appendix/glossary.md#g-过拟合) | 评估与调优 | [验证集](../appendix/glossary.md#g-验证集)、[泛化](../appendix/glossary.md#g-泛化) |
    | [正则化](../appendix/glossary.md#g-正则化) | 评估与调优 | [过拟合](../appendix/glossary.md#g-过拟合) |
    | [Dropout](../appendix/glossary.md#g-Dropout) | 评估与调优 | [正则化](../appendix/glossary.md#g-正则化) |
    | [权重衰减](../appendix/glossary.md#g-权重衰减) | 评估与调优 | [正则化](../appendix/glossary.md#g-正则化)、[权重](../appendix/glossary.md#g-权重) |
    | [数据增强](../appendix/glossary.md#g-数据增强) | 评估与调优 | [过拟合](../appendix/glossary.md#g-过拟合) |
    | [早停](../appendix/glossary.md#g-早停) | 评估与调优 | [验证集](../appendix/glossary.md#g-验证集)、[轮次](../appendix/glossary.md#g-轮次) |
    | [网格搜索](../appendix/glossary.md#g-网格搜索) | 评估与调优 | [超参数](../appendix/glossary.md#g-超参数)、[验证集](../appendix/glossary.md#g-验证集) |

??? note "第 5 章 卷积神经网络（12 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [通道](../appendix/glossary.md#g-通道) | 卷积与视觉 | [张量](../appendix/glossary.md#g-张量) |
    | [卷积](../appendix/glossary.md#g-卷积) | 卷积与视觉 | [通道](../appendix/glossary.md#g-通道) |
    | [卷积核](../appendix/glossary.md#g-卷积核) | 卷积与视觉 | [卷积](../appendix/glossary.md#g-卷积)、[权重](../appendix/glossary.md#g-权重) |
    | [特征图](../appendix/glossary.md#g-特征图) | 卷积与视觉 | [卷积核](../appendix/glossary.md#g-卷积核) |
    | [局部连接](../appendix/glossary.md#g-局部连接) | 卷积与视觉 | [卷积](../appendix/glossary.md#g-卷积) |
    | [权值共享](../appendix/glossary.md#g-权值共享) | 卷积与视觉 | [卷积核](../appendix/glossary.md#g-卷积核) |
    | [填充](../appendix/glossary.md#g-填充) | 卷积与视觉 | [卷积](../appendix/glossary.md#g-卷积) |
    | [步幅](../appendix/glossary.md#g-步幅) | 卷积与视觉 | [卷积](../appendix/glossary.md#g-卷积) |
    | [池化](../appendix/glossary.md#g-池化) | 卷积与视觉 | [特征图](../appendix/glossary.md#g-特征图) |
    | [最大池化](../appendix/glossary.md#g-最大池化) | 卷积与视觉 | [池化](../appendix/glossary.md#g-池化) |
    | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络) | 卷积与视觉 | [局部连接](../appendix/glossary.md#g-局部连接)、[权值共享](../appendix/glossary.md#g-权值共享)、[池化](../appendix/glossary.md#g-池化)、[隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [ResNet](../appendix/glossary.md#g-ResNet) | 卷积与视觉 | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络)、[残差连接](../appendix/glossary.md#g-残差连接) |

??? note "第 6 章 RNN 到 Transformer（13 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [序列数据](../appendix/glossary.md#g-序列数据) | 序列与语言 | [张量](../appendix/glossary.md#g-张量) |
    | [循环神经网络](../appendix/glossary.md#g-循环神经网络) | 序列与语言 | [序列数据](../appendix/glossary.md#g-序列数据)、[隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [隐藏状态](../appendix/glossary.md#g-隐藏状态) | 序列与语言 | [循环神经网络](../appendix/glossary.md#g-循环神经网络) |
    | [LSTM](../appendix/glossary.md#g-LSTM) | 序列与语言 | [隐藏状态](../appendix/glossary.md#g-隐藏状态)、[梯度消失](../appendix/glossary.md#g-梯度消失) |
    | [记忆细胞](../appendix/glossary.md#g-记忆细胞) | 序列与语言 | [LSTM](../appendix/glossary.md#g-LSTM) |
    | [遗忘门](../appendix/glossary.md#g-遗忘门) | 序列与语言 | [记忆细胞](../appendix/glossary.md#g-记忆细胞)、[Sigmoid](../appendix/glossary.md#g-Sigmoid) |
    | [输入门](../appendix/glossary.md#g-输入门) | 序列与语言 | [记忆细胞](../appendix/glossary.md#g-记忆细胞) |
    | [输出门](../appendix/glossary.md#g-输出门) | 序列与语言 | [记忆细胞](../appendix/glossary.md#g-记忆细胞) |
    | [注意力](../appendix/glossary.md#g-注意力) | 序列与语言 | [Softmax](../appendix/glossary.md#g-Softmax)、[循环神经网络](../appendix/glossary.md#g-循环神经网络) |
    | [自注意力](../appendix/glossary.md#g-自注意力) | 序列与语言 | [注意力](../appendix/glossary.md#g-注意力) |
    | [多头注意力](../appendix/glossary.md#g-多头注意力) | 序列与语言 | [自注意力](../appendix/glossary.md#g-自注意力) |
    | [位置编码](../appendix/glossary.md#g-位置编码) | 序列与语言 | [自注意力](../appendix/glossary.md#g-自注意力) |
    | [Transformer](../appendix/glossary.md#g-Transformer) | 序列与语言 | [多头注意力](../appendix/glossary.md#g-多头注意力)、[位置编码](../appendix/glossary.md#g-位置编码)、[残差连接](../appendix/glossary.md#g-残差连接) |

??? note "第 7 章 图像分类与迁移学习（12 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [图像分类](../appendix/glossary.md#g-图像分类) | 卷积与视觉 | [卷积神经网络](../appendix/glossary.md#g-卷积神经网络) |
    | [目标检测](../appendix/glossary.md#g-目标检测) | 卷积与视觉 | [图像分类](../appendix/glossary.md#g-图像分类) |
    | [图像分割](../appendix/glossary.md#g-图像分割) | 卷积与视觉 | [图像分类](../appendix/glossary.md#g-图像分类) |
    | [迁移学习](../appendix/glossary.md#g-迁移学习) | 卷积与视觉 | [预训练模型](../appendix/glossary.md#g-预训练模型) |
    | [预训练模型](../appendix/glossary.md#g-预训练模型) | 卷积与视觉 | [ImageNet](../appendix/glossary.md#g-ImageNet)、[ResNet](../appendix/glossary.md#g-ResNet) |
    | [ImageNet](../appendix/glossary.md#g-ImageNet) | 卷积与视觉 | [图像分类](../appendix/glossary.md#g-图像分类) |
    | [冻结](../appendix/glossary.md#g-冻结) | 卷积与视觉 | [迁移学习](../appendix/glossary.md#g-迁移学习) |
    | [微调](../appendix/glossary.md#g-微调) | 卷积与视觉 | [冻结](../appendix/glossary.md#g-冻结)、[学习率](../appendix/glossary.md#g-学习率) |
    | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) | 评估与调优 | [准确率](../appendix/glossary.md#g-准确率)、[测试集](../appendix/glossary.md#g-测试集) |
    | [精确率](../appendix/glossary.md#g-精确率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [召回率](../appendix/glossary.md#g-召回率) | 评估与调优 | [混淆矩阵](../appendix/glossary.md#g-混淆矩阵) |
    | [F1 值](../appendix/glossary.md#g-F1-值) | 评估与调优 | [精确率](../appendix/glossary.md#g-精确率)、[召回率](../appendix/glossary.md#g-召回率) |

??? note "第 8 章 文本分类与预训练（11 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [分词](../appendix/glossary.md#g-分词) | 序列与语言 | [序列数据](../appendix/glossary.md#g-序列数据) |
    | [词表](../appendix/glossary.md#g-词表) | 序列与语言 | [分词](../appendix/glossary.md#g-分词)、[训练集](../appendix/glossary.md#g-训练集) |
    | [补齐](../appendix/glossary.md#g-补齐) | 序列与语言 | [词表](../appendix/glossary.md#g-词表) |
    | [独热编码](../appendix/glossary.md#g-独热编码) | 序列与语言 | [词表](../appendix/glossary.md#g-词表) |
    | [词嵌入](../appendix/glossary.md#g-词嵌入) | 序列与语言 | [独热编码](../appendix/glossary.md#g-独热编码)、[参数](../appendix/glossary.md#g-参数) |
    | [词向量](../appendix/glossary.md#g-词向量) | 序列与语言 | [词嵌入](../appendix/glossary.md#g-词嵌入) |
    | [Word2Vec](../appendix/glossary.md#g-Word2Vec) | 序列与语言 | [词向量](../appendix/glossary.md#g-词向量) |
    | [余弦相似度](../appendix/glossary.md#g-余弦相似度) | 序列与语言 | [词向量](../appendix/glossary.md#g-词向量) |
    | [情感分类](../appendix/glossary.md#g-情感分类) | 序列与语言 | [词嵌入](../appendix/glossary.md#g-词嵌入)、[LSTM](../appendix/glossary.md#g-LSTM)、[精确率](../appendix/glossary.md#g-精确率) |
    | [预训练语言模型](../appendix/glossary.md#g-预训练语言模型) | 序列与语言 | [Transformer](../appendix/glossary.md#g-Transformer)、[迁移学习](../appendix/glossary.md#g-迁移学习) |
    | [BERT](../appendix/glossary.md#g-BERT) | 序列与语言 | [预训练语言模型](../appendix/glossary.md#g-预训练语言模型)、[微调](../appendix/glossary.md#g-微调) |

??? note "第 9 章 生成模型（13 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [判别模型](../appendix/glossary.md#g-判别模型) | 生成模型 | [图像分类](../appendix/glossary.md#g-图像分类) |
    | [生成模型](../appendix/glossary.md#g-生成模型) | 生成模型 | [判别模型](../appendix/glossary.md#g-判别模型) |
    | [自编码器](../appendix/glossary.md#g-自编码器) | 生成模型 | [编码器](../appendix/glossary.md#g-编码器)、[解码器](../appendix/glossary.md#g-解码器)、[均方误差](../appendix/glossary.md#g-均方误差) |
    | [编码器](../appendix/glossary.md#g-编码器) | 生成模型 | [隐藏层](../appendix/glossary.md#g-隐藏层) |
    | [解码器](../appendix/glossary.md#g-解码器) | 生成模型 | [编码器](../appendix/glossary.md#g-编码器) |
    | [潜在空间](../appendix/glossary.md#g-潜在空间) | 生成模型 | [自编码器](../appendix/glossary.md#g-自编码器) |
    | [异常检测](../appendix/glossary.md#g-异常检测) | 生成模型 | [自编码器](../appendix/glossary.md#g-自编码器) |
    | [生成对抗网络](../appendix/glossary.md#g-生成对抗网络) | 生成模型 | [生成模型](../appendix/glossary.md#g-生成模型)、[交叉熵](../appendix/glossary.md#g-交叉熵) |
    | [生成器](../appendix/glossary.md#g-生成器) | 生成模型 | [生成对抗网络](../appendix/glossary.md#g-生成对抗网络) |
    | [判别器](../appendix/glossary.md#g-判别器) | 生成模型 | [生成对抗网络](../appendix/glossary.md#g-生成对抗网络) |
    | [模式崩溃](../appendix/glossary.md#g-模式崩溃) | 生成模型 | [生成器](../appendix/glossary.md#g-生成器) |
    | [扩散模型](../appendix/glossary.md#g-扩散模型) | 生成模型 | [生成模型](../appendix/glossary.md#g-生成模型)、[自编码器](../appendix/glossary.md#g-自编码器) |
    | [深度伪造](../appendix/glossary.md#g-深度伪造) | 强化学习与责任 | [扩散模型](../appendix/glossary.md#g-扩散模型)、[生成对抗网络](../appendix/glossary.md#g-生成对抗网络) |

??? note "第 10 章 强化学习与综合项目（14 个概念）"

    | 概念 | 类别 | 直接先修 |
    |---|---|---|
    | [强化学习](../appendix/glossary.md#g-强化学习) | 强化学习与责任 | [神经网络](../appendix/glossary.md#g-神经网络) |
    | [智能体](../appendix/glossary.md#g-智能体) | 强化学习与责任 | [强化学习](../appendix/glossary.md#g-强化学习) |
    | [环境](../appendix/glossary.md#g-环境) | 强化学习与责任 | [强化学习](../appendix/glossary.md#g-强化学习) |
    | [状态](../appendix/glossary.md#g-状态) | 强化学习与责任 | [环境](../appendix/glossary.md#g-环境) |
    | [动作](../appendix/glossary.md#g-动作) | 强化学习与责任 | [智能体](../appendix/glossary.md#g-智能体) |
    | [奖励](../appendix/glossary.md#g-奖励) | 强化学习与责任 | [动作](../appendix/glossary.md#g-动作) |
    | [策略](../appendix/glossary.md#g-策略) | 强化学习与责任 | [状态](../appendix/glossary.md#g-状态)、[动作](../appendix/glossary.md#g-动作) |
    | [Q 值](../appendix/glossary.md#g-Q-值) | 强化学习与责任 | [奖励](../appendix/glossary.md#g-奖励)、[策略](../appendix/glossary.md#g-策略) |
    | [DQN](../appendix/glossary.md#g-DQN) | 强化学习与责任 | [Q 值](../appendix/glossary.md#g-Q-值)、[神经网络](../appendix/glossary.md#g-神经网络) |
    | [ε-贪心](../appendix/glossary.md#g-ε-贪心) | 强化学习与责任 | [DQN](../appendix/glossary.md#g-DQN) |
    | [经验回放](../appendix/glossary.md#g-经验回放) | 强化学习与责任 | [DQN](../appendix/glossary.md#g-DQN) |
    | [目标网络](../appendix/glossary.md#g-目标网络) | 强化学习与责任 | [DQN](../appendix/glossary.md#g-DQN) |
    | [偏见](../appendix/glossary.md#g-偏见) | 强化学习与责任 | [训练集](../appendix/glossary.md#g-训练集)、[精确率](../appendix/glossary.md#g-精确率) |
    | [基准方法](../appendix/glossary.md#g-基准方法) | 强化学习与责任 | [测试集](../appendix/glossary.md#g-测试集) |

<!-- graph:end -->
