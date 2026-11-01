---
title: 期末模拟卷
description: 按期末开卷考试的题型和分值，从各章课后练习中组成的两套在线模拟卷，可以直接作答并查看解析
---

# 期末模拟卷

本课程期末为开卷考试，题目侧重理解和应用。下面两套模拟卷按期末考试的题型和分值，从各章课后练习中抽题组成，覆盖全书 10 章：

| 题型 | 题数 | 每题分值 | 合计 |
|---|---|---|---|
| 单项选择题 | 20 | 2 | 40 |
| 多项选择题 | 5 | 4 | 20 |
| 判断题 | 10 | 1 | 10 |
| 简答题 | 2 | 10 | 20 |
| 实操题 | 1 | 10 | 10 |

建议限时 90 分钟，先独立做完一整套，再展开“答案与解析”核对。简答题和实操题按参考答案的要点自评。实操题要在电脑上运行代码完成。选择题和判断题可以直接点击作答；每道题解析下方的“回看”链接可以跳回对应章节复习。

## 模拟卷一

**一、单项选择题**（每题 2 分）

<div class="quiz" data-answer="C" markdown>

**1.** 一个神经元的输入 x = (2, 1)，权重 w = (0.5, −1)，偏置 b = 0.5，激活函数为 ReLU。它的输出是（　　）。

- A. 0
- B. 1.5
- C. 0.5
- D. −0.5

</div>

??? tip "答案与解析"
    **答案：C**

    z = 0.5 × 2 + (−1) × 1 + 0.5 = 0.5，大于 0，ReLU 原样输出 0.5。B 是把 −1 × 1 当成 +1 算错的结果；D 忘了加偏置（1 − 1 − 0.5 当成了减偏置）。

    考查：人工神经元、ReLU · 认知层次：应用 · [回看 1.2.2](chapters/01-neuron/index.md#122)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**2.** 三个样本的预测值为 (2, 4, 6)，真实值为 (3, 4, 4)，均方误差是（　　）。

- A. 1
- B. 5/3（约 1.67）
- C. 3
- D. 5

</div>

??? tip "答案与解析"
    **答案：B**

    差为 −1、0、2，平方为 1、0、4，平均 (1 + 0 + 4) / 3 = 5/3。D 是忘了除以 3；C 是用差的绝对值之和。

    考查：均方误差 · 认知层次：应用 · [回看 2.1.1](chapters/02-training/index.md#211)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**3.** 反向传播的作用是（　　）。

- A. 从输入算到输出，得到预测值
- B. 从损失出发往回推，算出每个参数的梯度
- C. 直接把参数更新为最优值
- D. 把训练数据打乱顺序

</div>

??? tip "答案与解析"
    **答案：B**

    A 是前向传播；C 是优化器按梯度更新参数，而且只是“走一步”，不是一步到最优；D 是 DataLoader 的工作。

    考查：反向传播 · 认知层次：记忆 · [回看 3.1](chapters/03-backprop/index.md#31)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**4.** 训练过程中用来比较不同超参数、决定何时停止训练的数据集是（　　）。

- A. 训练集
- B. 验证集
- C. 测试集
- D. 任何一个都可以

</div>

??? tip "答案与解析"
    **答案：B**

    验证集好比模拟考，用来做选择；测试集只在最后考一次；训练集用来更新参数。

    考查：验证集 · 认知层次：记忆 · [回看 4.1.1](chapters/04-generalization/index.md#411)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**5.** 一张 224×224 的彩色照片摊平后有多少个数？（　　）

- A. 50 176
- B. 150 528
- C. 224
- D. 672

</div>

??? tip "答案与解析"
    **答案：B**

    224 × 224 × 3 = 150 528，彩色照片有红、绿、蓝 3 个通道。

    考查：通道 · 认知层次：应用 · [回看 5.1.1](chapters/05-cnn/index.md#511)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**6.** 下列数据中，**不属于**序列数据的是（　　）。

- A. 一段录音
- B. 一个月每天的销量
- C. 一张证件照
- D. 一条微博评论

</div>

??? tip "答案与解析"
    **答案：C**

    一张照片是“面”状数据，用 CNN 处理；其他三种都有先后顺序。

    考查：序列数据 · 认知层次：理解 · [回看 6.1.1](chapters/06-sequence/index.md#611)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**7.** 工地上要用摄像头找出没戴安全帽的工人，并标出他们在画面中的位置，这属于（　　）。

- A. 图像分类
- B. 目标检测
- C. 图像分割
- D. 图像生成

</div>

??? tip "答案与解析"
    **答案：B**

    既要知道“是什么”，又要知道“在哪里”（画方框），是目标检测。

    考查：计算机视觉任务 · 认知层次：理解 · [回看 7.1](chapters/07-image-transfer/index.md#71)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**8.** 中文文本处理的第一步通常是分词，主要原因是（　　）。

- A. 中文的词和词之间没有空格
- B. 中文的字太多
- C. 网络只能处理英文
- D. 为了减少标点符号

</div>

??? tip "答案与解析"
    **答案：A**

    英文天然以空格分隔单词，中文需要专门的分词工具（如 jieba）切分。

    考查：分词 · 认知层次：理解 · [回看 8.1.1](chapters/08-text/index.md#811)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**9.** 下列任务中，属于生成模型应用的是（　　）。

- A. 判断一封邮件是不是垃圾邮件
- B. 根据文字描述画一幅画
- C. 识别照片中的车牌号
- D. 预测明天的气温

</div>

??? tip "答案与解析"
    **答案：B**

    生成模型输出新的图片、文字等数据；其余三项都是判断或预测。

    考查：判别与生成 · 认知层次：理解 · [回看 9.1](chapters/09-generative/index.md#91)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**10.** 强化学习与监督学习最主要的区别是（　　）。

- A. 强化学习不用神经网络
- B. 强化学习只能用于下棋
- C. 强化学习没有每一步的标准答案，只有做完后得到的奖励
- D. 强化学习不需要计算机

</div>

??? tip "答案与解析"
    **答案：C**

    强化学习在试错中学习，靠奖励来判断好坏。

    考查：强化学习 · 认知层次：理解 · [回看 10.1](chapters/10-drl-project/index.md#101)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**11.** Sigmoid(0) 的值是（　　）。

- A. 0
- B. 0.5
- C. 1
- D. 无法计算

</div>

??? tip "答案与解析"
    **答案：B**

    Sigmoid(0) = 1 / (1 + e⁰) = 1 / 2 = 0.5。Sigmoid 的输出总在 0 和 1 之间，z = 0 时正好在中间。

    考查：Sigmoid · 认知层次：应用 · [回看 1.3.1](chapters/01-neuron/index.md#131)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**12.** 识别手写数字（10 类）应该选用哪种损失函数？（　　）

- A. 均方误差
- B. 准确率
- C. 交叉熵
- D. 学习率

</div>

??? tip "答案与解析"
    **答案：C**

    分类任务用交叉熵。准确率是评估指标，它不能求梯度，不能直接用来训练；学习率是超参数。

    考查：交叉熵 · 认知层次：理解 · [回看 2.1.2](chapters/02-training/index.md#212)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**13.** y = (2x + 1)²，x = 2 时 dy/dx 是（　　）。

- A. 5
- B. 10
- C. 20
- D. 25

</div>

??? tip "答案与解析"
    **答案：C**

    u = 2 × 2 + 1 = 5，dy/du = 2u = 10，du/dx = 2，dy/dx = 10 × 2 = 20。D 是 y 本身的值；B 漏乘了 du/dx。

    考查：链式法则 · 认知层次：应用 · [回看 3.2.1](chapters/03-backprop/index.md#321)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**14.** 训练损失持续下降，验证损失先下降后上升，这说明模型（　　）。

- A. 欠拟合
- B. 学习率太小
- C. 过拟合
- D. 训练正常

</div>

??? tip "答案与解析"
    **答案：C**

    两条曲线分道扬镳是过拟合的典型特征。

    考查：过拟合 · 认知层次：分析 · [回看 4.2.2](chapters/04-generalization/index.md#422)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**15.** 图片 `[[1, 0], [0, 1]]` 与卷积核 `[[1, 2], [3, 4]]` 在唯一的位置上做卷积（不加偏置），结果是（　　）。

- A. 10
- B. 2
- C. 5
- D. 0

</div>

??? tip "答案与解析"
    **答案：C**

    对应位置相乘再相加：1×1 + 0×2 + 0×3 + 1×4 = 5。

    考查：卷积 · 认知层次：应用 · [回看 5.2.1](chapters/05-cnn/index.md#521)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**16.** RNN 读到第 t 个字时，计算新的隐藏状态要用到（　　）。

- A. 只有第 t 个字
- B. 第 t 个字和上一步的隐藏状态
- C. 整句话的所有字
- D. 下一个字

</div>

??? tip "答案与解析"
    **答案：B**

    新笔记 = tanh(W × 当前的字 + U × 旧笔记 + b)。

    考查：循环神经网络 · 认知层次：记忆 · [回看 6.1.2](chapters/06-sequence/index.md#612)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**17.** 迁移学习中，“冻结”是指（　　）。

- A. 把模型保存到硬盘
- B. 停止训练
- C. 让预训练模型的参数在训练中保持不变
- D. 把学习率设为 0.1

</div>

??? tip "答案与解析"
    **答案：C**

    冻结就是把参数的 `requires_grad` 设为 `False`，不计算梯度、不更新。

    考查：冻结 · 认知层次：记忆 · [回看 7.3.2](chapters/07-image-transfer/index.md#732)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**18.** 建词表时预留的“0 号”通常用来表示（　　）。

- A. 最常见的词
- B. 句号
- C. 补齐
- D. 好评

</div>

??? tip "答案与解析"
    **答案：C**

    0 号给补齐，1 号给“未知”（词表里没有的词）。`padding_idx=0` 让补齐的向量恒为 0。

    考查：建词表 · 认知层次：记忆 · [回看 8.1.2](chapters/08-text/index.md#812)
    { .quiz-meta }

<div class="quiz" data-answer="C" markdown>

**19.** 训练自编码器时，它的“标签”（目标）是（　　）。

- A. 图片的类别
- B. 随机数
- C. 输入图片本身
- D. 不需要目标

</div>

??? tip "答案与解析"
    **答案：C**

    自编码器要让输出尽量和输入一样，不需要人工标注。降噪自编码器的目标是干净的原图。

    考查：自编码器 · 认知层次：理解 · [回看 9.2.1](chapters/09-generative/index.md#921)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**20.** 在平衡小车问题中，“杆子没倒，每坚持一步得 1 分”属于强化学习的（　　）。

- A. 奖励
- B. 状态
- C. 动作
- D. 环境

</div>

??? tip "答案与解析"
    **答案：A**

    状态是 4 个数，动作是向左或向右推。

    考查：奖励 · 认知层次：记忆 · [回看 10.1](chapters/10-drl-project/index.md#101)
    { .quiz-meta }

**二、多项选择题**（每题 4 分）

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**21.** 一个人工神经元在计算时会做哪些事？（　　）

- A. 把每个输入乘上对应的权重
- B. 把乘积加起来，再加上偏置
- C. 把结果和训练集中的所有样本比较
- D. 把加权和送进激活函数

</div>

??? tip "答案与解析"
    **答案：ABD**

    “乘一乘、加一加、拐个弯”。C 不是神经元做的事。

    考查：人工神经元 · 认知层次：记忆 · [回看 1.2.2](chapters/01-neuron/index.md#122)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABCD" markdown>

**22.** 下列哪些方法可以缓解深层网络的梯度消失？（　　）

- A. 隐藏层用 ReLU 代替 Sigmoid
- B. 使用适合 ReLU 的初始化方法
- C. 加入批归一化
- D. 加入残差连接

</div>

??? tip "答案与解析"
    **答案：ABCD**

    四种都是常用对策，实验 3-2 验证了 B、C 的效果。

    考查：梯度消失 · 认知层次：理解 · [回看 3.4.2](chapters/03-backprop/index.md#342)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**23.** 关于池化层，下列说法正确的有（　　）。

- A. 2×2 最大池化把特征图的高和宽各减半
- B. 池化层没有需要训练的参数
- C. 池化层会改变通道数
- D. 池化能让网络容忍图案的小幅位置变化

</div>

??? tip "答案与解析"
    **答案：ABD**

    池化在每个通道上分别进行，通道数不变。

    考查：池化 · 认知层次：理解 · [回看 5.3.1](chapters/05-cnn/index.md#531)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABC" markdown>

**24.** 迁移学习有效的原因包括（　　）。

- A. CNN 前面的层学到的边缘、纹理等特征是通用的
- B. 预训练模型在大量图片上训练过，“看图本领”强
- C. 只需训练少量参数，数据少也不容易过拟合
- D. 预训练模型一定见过你要识别的所有类别

</div>

??? tip "答案与解析"
    **答案：ABC**

    预训练模型不一定见过你的类别（比如工厂的零件），但它学到的通用特征依然有用。

    考查：迁移学习 · 认知层次：理解 · [回看 7.3.1](chapters/07-image-transfer/index.md#731)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**25.** 自编码器可以用于（　　）。

- A. 给图片降噪
- B. 异常检测
- C. 直接判断评论是好评还是差评
- D. 数据压缩与降维

</div>

??? tip "答案与解析"
    **答案：ABD**

    自编码器本身不做分类；编码可以作为特征再交给分类器。

    考查：自编码器 · 认知层次：记忆 · [回看 9.2.2](chapters/09-generative/index.md#922)
    { .quiz-meta }

**三、判断题**（每题 1 分）

<div class="quiz quiz-judge" data-answer="×" markdown>

**26.** 神经网络的权重是程序员根据经验一个个写进程序的。

</div>

??? tip "答案与解析"
    **答案：×**

    权重和偏置是通过训练从数据中学出来的，程序员只决定网络的结构。

    考查：参数、训练 · 认知层次：理解 · [回看 1.2.3](chapters/01-neuron/index.md#123)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**27.** 损失函数的值越小，说明网络的预测和真实答案越接近。

</div>

??? tip "答案与解析"
    **答案：√**

    损失函数就是衡量差距的“尺子”。

    考查：损失函数 · 认知层次：记忆 · [回看 2.1](chapters/02-training/index.md#21)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**28.** 反向传播要用到前向传播时算出的中间结果，所以必须先做前向传播。

</div>

??? tip "答案与解析"
    **答案：√**

    例如算 dy/du = 2u 时需要知道前向算出的 u。

    考查：反向传播 · 认知层次：理解 · [回看 3.2.1](chapters/03-backprop/index.md#321)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**29.** 模型在训练集上的准确率达到 100%，说明它已经训练得非常好了。

</div>

??? tip "答案与解析"
    **答案：×**

    训练集 100% 可能只是“背下来了”，要看验证集上的表现。

    考查：泛化 · 认知层次：评价 · [回看 4.1.1](chapters/04-generalization/index.md#411)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**30.** 卷积神经网络里卷积核的数值，需要工程师根据经验手工设计。

</div>

??? tip "答案与解析"
    **答案：×**

    卷积核的数值和全连接层的权重一样，是通过反向传播和梯度下降训练出来的。实验 5-1 手工设置卷积核只是为了演示。

    考查：卷积核 · 认知层次：理解 · [回看 5.2.2](chapters/05-cnn/index.md#522)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**31.** RNN 在每个时刻使用同一套参数，所以参数个数与序列长度无关。

</div>

??? tip "答案与解析"
    **答案：√**

    这与卷积核在图上到处使用同一套参数的道理相同。

    考查：循环神经网络 · 认知层次：理解 · [回看 6.1.2](chapters/06-sequence/index.md#612)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**32.** 图像分类、目标检测、图像分割三类任务中，图像分类最简单，也是另外两类的基础。

</div>

??? tip "答案与解析"
    **答案：√**

    检测和分割通常以分类网络为“骨干”提取特征。

    考查：计算机视觉任务 · 认知层次：理解 · [回看 7.1](chapters/07-image-transfer/index.md#71)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**33.** 建词表时应该把训练集和测试集的所有词都统计进去，这样未知的词更少。

</div>

??? tip "答案与解析"
    **答案：×**

    词表只能用训练集来建，否则就偷看了测试数据。

    考查：建词表 · 认知层次：理解 · [回看 8.1.2](chapters/08-text/index.md#812)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**34.** 训练自编码器不需要人工标注的标签。

</div>

??? tip "答案与解析"
    **答案：√**

    图片自己就是答案。

    考查：自编码器 · 认知层次：理解 · [回看 9.2.1](chapters/09-generative/index.md#921)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**35.** 强化学习的目标是让累计得到的奖励最大，而不只是眼前这一步的奖励最大。

</div>

??? tip "答案与解析"
    **答案：√**

    就像下棋时先弃一个子，后面赢一大块。

    考查：强化学习 · 认知层次：理解 · [回看 10.1](chapters/10-drl-project/index.md#101)
    { .quiz-meta }

**四、简答题**（每题 10 分）

**36.** 用“蒙眼下山”的比喻解释梯度下降，并说明学习率太大和太小各会出现什么现象。

??? tip "答案与解析"
    **参考答案**：蒙眼站在山坡上，看不到整座山，只能感觉脚下哪边往下；每次朝最陡的下坡方向迈一小步，再重新感觉，反复多次就能走到谷底。山的高度就是损失，脚下的坡度就是梯度，每一步迈多远就是学习率。学习率太小，每步只挪一点点，走很久都到不了谷底，损失下降很慢；学习率太大，一步迈过谷底跳到对面，损失来回跳甚至越来越大（发散）。

    评分要点：比喻对应关系 2 分，两种现象各 1 分。

    考查：梯度下降、学习率 · 认知层次：理解 · [回看 2.2](chapters/02-training/index.md#22)
    { .quiz-meta }

**37.** 用“重点笔记本”的比喻，说明 LSTM 为什么比简单 RNN 记得更久。

??? tip "答案与解析"
    **参考答案**：简单 RNN 只有一份笔记，每读一个字就整份改写一遍，旧内容一次次被改动，传得越远越走样。LSTM 另有一本“重点笔记本”（记忆细胞），由三道门管理：遗忘门决定擦掉哪些旧内容，输入门决定记下哪些新内容，输出门决定这一步拿出哪些内容来用。遇到不重要的字时，遗忘门接近 1、输入门接近 0，笔记本几乎原样往后传，开头记下的重要内容（如“成都”）就能保留到很远的地方。

    考查：LSTM · 认知层次：理解 · [回看 6.3](chapters/06-sequence/index.md#63)
    { .quiz-meta }

**五、实操题**（每题 10 分）

**38.** 设 L = (w·x + b − y)²，其中 x = 3、y = 10，初始 w = 1、b = 1。（1）手算 dL/dw 和 dL/db；（2）用 PyTorch 自动求导验证；（3）学习率 0.01，更新一步后损失是多少？

??? tip "答案与解析"
    **参考答案**：（1）前向 z = 3 + 1 = 4，L = 36；dL/dz = 2(4 − 10) = −12，dL/dw = −12 × 3 = −36，dL/db = −12。（2）关键代码：

    ```python
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    loss = (w * 3 + b - 10) ** 2
    loss.backward()
    print(w.grad, b.grad)        # tensor(-36.) tensor(-12.)
    ```

    （3）w = 1 + 0.36 = 1.36，b = 1 + 0.12 = 1.12；z = 1.36 × 3 + 1.12 = 5.2，L = (5.2 − 10)² = 23.04，从 36 降到 23.04。

    考查：链式法则、自动求导 · 认知层次：应用 · [回看 3.6](chapters/03-backprop/index.md#36)
    { .quiz-meta }

## 模拟卷二

**一、单项选择题**（每题 2 分）

<div class="quiz" data-answer="D" markdown>

**1.** 把一个多层感知机所有隐藏层的激活函数都去掉，会发生什么？（　　）

- A. 训练更快，效果更好
- B. 网络只能输出 0 或 1
- C. 网络的参数个数减少一半
- D. 不管多少层，整个网络都等价于一层，只能画出直线分界线

</div>

??? tip "答案与解析"
    **答案：D**

    两个线性运算接起来还是线性运算，例如 3(2x + 1) − 4 = 6x − 1。激活函数没有参数，去掉它不改变参数个数，C 错。

    考查：激活函数、非线性 · 认知层次：理解 · [回看 1.3.2](chapters/01-neuron/index.md#132)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**2.** 一个 10 分类网络刚开始训练时在“瞎猜”，每类概率都约为 0.1。此时它的交叉熵损失大约是（　　）。

- A. 2.3
- B. 0.1
- C. 0
- D. 10

</div>

??? tip "答案与解析"
    **答案：A**

    交叉熵 = −ln 0.1 ≈ 2.303。这是判断“网络有没有开始学”的一个参照值。

    考查：交叉熵 · 认知层次：应用 · [回看 2.1.2](chapters/02-training/index.md#212)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**3.** 一个神经元 z = w·x + b，损失 L = (z − y)²。若 w = 1、b = 0、x = 3、y = 1，则 dL/dw 是（　　）。

- A. 12
- B. 4
- C. 6
- D. −12

</div>

??? tip "答案与解析"
    **答案：A**

    z = 3，dL/dz = 2(3 − 1) = 4，dL/dw = 4 × x = 12。B 是 dL/dz（也等于 dL/db），忘了乘 x。

    考查：链式法则 · 认知层次：应用 · [回看 3.2.2](chapters/03-backprop/index.md#322)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**4.** 训练了 50 轮，训练损失和验证损失都停在 1.5 左右，几乎不再下降，而且两者很接近。最合适的改进是（　　）。

- A. 加大网络或调整学习率
- B. 加 Dropout
- C. 加权重衰减
- D. 提前停止训练

</div>

??? tip "答案与解析"
    **答案：A**

    这是欠拟合，模型能力不够或没学好；B、C、D 都是治过拟合的，用在这里会更糟。

    考查：欠拟合 · 认知层次：评价 · [回看 4.2.2](chapters/04-generalization/index.md#422)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**5.** 特征图上所有位置都使用同一个卷积核，这叫做（　　）。

- A. 权值共享
- B. 局部连接
- C. 池化
- D. 填充

</div>

??? tip "答案与解析"
    **答案：A**

    权值共享让参数个数与图片大小无关，也让同一图案在任何位置都能被找到。局部连接指每个输出只看一小块区域。

    考查：权值共享 · 认知层次：记忆 · [回看 5.2.2](chapters/05-cnn/index.md#522)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**6.** 实验 6-1 中，64 个样本、每个样本 30 天、每天 1 个气温值，送入 `nn.LSTM(..., batch_first=True)` 的输入形状是（　　）。

- A. 64×30×1
- B. 30×64×1
- C. 64×1×30
- D. 64×30

</div>

??? tip "答案与解析"
    **答案：A**

    `batch_first=True` 时形状为“批量 × 步数 × 每步的特征数”。D 少了最后一维，会报错。

    考查：循环神经网络 · 认知层次：应用 · [回看 6.1.3](chapters/06-sequence/index.md#613)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**7.** 解冻预训练模型进行微调时，学习率通常应该（　　）。

- A. 比冻结训练时小得多
- B. 比冻结训练时大得多
- C. 和从零训练时一样
- D. 越大越好

</div>

??? tip "答案与解析"
    **答案：A**

    学习率太大会破坏预训练学到的特征。实验 7-2 微调用 0.0001，是冻结时的十分之一。

    考查：微调 · 认知层次：理解 · [回看 7.3.2](chapters/07-image-transfer/index.md#732)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**8.** 下列关于独热编码的说法，正确的是（　　）。

- A. 意思相近的词，独热向量也相近
- B. 任意两个词的独热向量互相垂直，看不出词的相近程度
- C. 独热向量很短
- D. 独热向量的值是训练出来的

</div>

??? tip "答案与解析"
    **答案：B**

    独热向量长度等于词表大小，只有一个位置是 1，任意两个不同词的点积都是 0。

    考查：独热编码 · 认知层次：理解 · [回看 8.1.3](chapters/08-text/index.md#813)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**9.** 自编码器中间的编码只有很少几个数，作用是（　　）。

- A. 迫使网络只保留最重要的信息
- B. 让训练更快
- C. 减少输入图片的像素
- D. 让网络能做分类

</div>

??? tip "答案与解析"
    **答案：A**

    “瓶颈”太窄，网络无法记住每个像素，只能学会抓住关键特征。

    考查：编码 · 认知层次：理解 · [回看 9.2.1](chapters/09-generative/index.md#921)
    { .quiz-meta }

<div class="quiz" data-answer="B" markdown>

**10.** Q 值表示（　　）。

- A. 这一步得到的奖励
- B. 在某个状态下做某个动作，预计从现在到最后总共能得多少分
- C. 网络的参数个数
- D. 智能体已经玩了多少局

</div>

??? tip "答案与解析"
    **答案：B**

    Q 值考虑的是累计奖励，不只是眼前这一步。

    考查：Q 值 · 认知层次：记忆 · [回看 10.1](chapters/10-drl-project/index.md#101)
    { .quiz-meta }

<div class="quiz" data-answer="A" markdown>

**11.** PyTorch 中的 `nn.Linear(10, 5)` 表示（　　）。

- A. 输入 10 个特征、输出 5 个特征的全连接层，共 55 个参数
- B. 输入 10 个特征、输出 5 个特征的全连接层，共 50 个参数
- C. 10 个神经元排成 5 层
- D. 输入 5 个特征、输出 10 个特征的全连接层

</div>

??? tip "答案与解析"
    **答案：A**

    `nn.Linear(进, 出)`，参数 = 10 × 5 + 5 = 55。B 漏了 5 个偏置。

    考查：全连接层、参数个数 · 认知层次：应用 · [回看 1.5.1](chapters/01-neuron/index.md#151)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**12.** 梯度下降的更新规则是（　　）。

- A. 新参数 = 旧参数 + 学习率 × 梯度
- B. 新参数 = 学习率 × 梯度
- C. 新参数 = 旧参数 × 学习率
- D. 新参数 = 旧参数 − 学习率 × 梯度

</div>

??? tip "答案与解析"
    **答案：D**

    减号表示朝梯度的反方向，也就是下坡方向走。A 是往上坡走，损失会变大。

    考查：梯度下降 · 认知层次：记忆 · [回看 2.2.1](chapters/02-training/index.md#221)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**13.** 下列关于 PyTorch 自动求导的说法，错误的是（　　）。

- A. 需要求梯度的张量要设置 `requires_grad=True`
- B. 调用 `loss.backward()` 后，梯度保存在参数的 `.grad` 里
- C. 连续两次 `backward()`，梯度会累加
- D. 评估模型时也必须记录计算图，否则无法得到预测结果

</div>

??? tip "答案与解析"
    **答案：D**

    评估时不需要梯度，应写 `with torch.no_grad():` 关掉计算图记录，预测结果照样能得到，而且更省内存。

    考查：自动求导 · 认知层次：理解 · [回看 3.3](chapters/03-backprop/index.md#33)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**14.** 关于 Dropout，下列说法正确的是（　　）。

- A. 训练和评估时都随机让一部分神经元休息
- B. 它会永久删除一部分神经元
- C. 它主要用来加快训练速度
- D. 只在训练时起作用，评估前要调用 `model.eval()` 关掉它

</div>

??? tip "答案与解析"
    **答案：D**

    Dropout 每批重新随机挑选休息的神经元，评估时全部上岗；它的作用是减轻过拟合，训练反而会慢一些。

    考查：Dropout · 认知层次：理解 · [回看 4.3.1](chapters/04-generalization/index.md#431)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**15.** `nn.Conv2d(3, 16, kernel_size=3)` 的权重张量形状是（　　）。

- A. 3×16×3×3
- B. 16×3×3
- C. 3×3×16
- D. 16×3×3×3

</div>

??? tip "答案与解析"
    **答案：D**

    16 个卷积核，每个卷积核有 3 层（对应 3 个输入通道），每层 3×3。实验 5-1 中 `nn.Conv2d(1, 3, 3)` 的权重形状是 (3, 1, 3, 3)，规律相同。

    考查：通道 · 认知层次：理解 · [回看 5.2.3](chapters/05-cnn/index.md#523)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**16.** 简单 RNN 记不住长序列前面信息的主要原因是（　　）。

- A. 参数太多
- B. 没有使用 GPU
- C. 学习率太小
- D. 信息和梯度一步步反复相乘，越传越小

</div>

??? tip "答案与解析"
    **答案：D**

    与第 3 章的梯度消失道理相同，像传话游戏一样越传越走样。

    考查：梯度消失 · 认知层次：理解 · [回看 6.2](chapters/06-sequence/index.md#62)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**17.** 实验 7-2 中，冻结 ResNet18 后要训练的参数只有 5130 个，这是因为（　　）。

- A. ResNet18 本来就只有 5130 个参数
- B. 图片缩小到了 112×112
- C. 只用了 2400 张图片
- D. 只训练新换上的最后一层：512 × 10 + 10

</div>

??? tip "答案与解析"
    **答案：D**

    ResNet18 有 1169 万个参数，冻结后只剩最后一层（512 个输入、10 个输出）参与训练。

    考查：冻结 · 认知层次：应用 · [回看实验 7-2](chapters/07-image-transfer/index.md#7-2)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**18.** `nn.Embedding(5000, 64)` 的参数个数是（　　）。

- A. 5000
- B. 64
- C. 5064
- D. 320 000

</div>

??? tip "答案与解析"
    **答案：D**

    它是一张 5000 行、64 列的查找表，5000 × 64 = 320 000，没有偏置。

    考查：词嵌入 · 认知层次：应用 · [回看 8.1.3](chapters/08-text/index.md#813)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**19.** 用只在合格产品照片上训练的自编码器做质检，判断产品有缺陷的依据是（　　）。

- A. 编码的值全为 0
- B. 训练损失很小
- C. 解码器的参数很多
- D. 这张照片的还原误差特别大

</div>

??? tip "答案与解析"
    **答案：D**

    自编码器只会还原见过的“正常”样子，没见过的缺陷还原不好，误差就大。

    考查：异常检测 · 认知层次：应用 · [回看 9.2.2](chapters/09-generative/index.md#922)
    { .quiz-meta }

<div class="quiz" data-answer="D" markdown>

**20.** DQN 用神经网络代替 Q 表的主要原因是（　　）。

- A. 神经网络训练更快
- B. Q 表不能存小数
- C. 神经网络不需要奖励
- D. 状态太多（甚至无穷多），Q 表列不完

</div>

??? tip "答案与解析"
    **答案：D**

    网络还能对没见过的状态估计 Q 值。

    考查：DQN · 认知层次：理解 · [回看 10.2.1](chapters/10-drl-project/index.md#1021)
    { .quiz-meta }

**二、多项选择题**（每题 4 分）

<div class="quiz quiz-multi" data-answer="ACD" markdown>

**21.** 下列哪些是超参数（需要人事先设定，不是训练学出来的）？（　　）

- A. 学习率
- B. 神经元的权重
- C. 批大小
- D. 训练轮数

</div>

??? tip "答案与解析"
    **答案：ACD**

    权重是参数，通过训练学出来；学习率、批大小、轮数都是人定的超参数。

    考查：超参数 · 认知层次：理解 · [回看 2.2.2](chapters/02-training/index.md#222)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**22.** 下列哪些做法会造成数据泄露？（　　）

- A. 反复查看测试集成绩来挑选超参数
- B. 同一位病人的多张 X 光片，一部分在训练集，一部分在测试集
- C. 测试集只在最后评估一次
- D. 用全部数据（包括测试集）计算归一化用的平均值

</div>

??? tip "答案与解析"
    **答案：ABD**

    C 是正确做法。B 中同一个人的特征会被模型“记住”，测试成绩虚高。

    考查：数据泄露 · 认知层次：分析 · [回看 4.1.2](chapters/04-generalization/index.md#412)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**23.** 与 RNN 相比，Transformer 的优点有（　　）。

- A. 任意两个位置一步就能互相“看到”
- B. 可以并行计算，适合 GPU 大规模训练
- C. 在任何数据量下都比 LSTM 效果好
- D. 长距离的信息不容易丢失

</div>

??? tip "答案与解析"
    **答案：ABD**

    数据量小时 Transformer 容易过拟合，未必比 LSTM 好（实验 6-3 中两者相近）。

    考查：Transformer · 认知层次：分析 · [回看 6.5](chapters/06-sequence/index.md#65)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABD" markdown>

**24.** 把中文评论变成网络能处理的数字，需要的步骤有（　　）。

- A. 分词
- B. 建词表并编号
- C. 把评论翻译成英文
- D. 补齐与截断到相同长度

</div>

??? tip "答案与解析"
    **答案：ABD**

    还有最后一步词嵌入，由网络中的 `nn.Embedding` 完成。

    考查：文本预处理 · 认知层次：记忆 · [回看 8.1](chapters/08-text/index.md#81)
    { .quiz-meta }

<div class="quiz quiz-multi" data-answer="ABC" markdown>

**25.** DQN 为了让训练稳定而采用的做法有（　　）。

- A. 经验回放
- B. 目标网络
- C. ε-贪心
- D. 把所有奖励都设为 0

</div>

??? tip "答案与解析"
    **答案：ABC**

    奖励全为 0 就什么也学不到。

    考查：DQN · 认知层次：记忆 · [回看 10.2.2](chapters/10-drl-project/index.md#1022)
    { .quiz-meta }

**三、判断题**（每题 1 分）

<div class="quiz quiz-judge" data-answer="√" markdown>

**26.** 前向传播是指从输入层开始，一层层算到输出层的过程。

</div>

??? tip "答案与解析"
    **答案：√**

    网络“使用”时做的就是一次前向传播。

    考查：前向传播 · 认知层次：记忆 · [回看 1.4.3](chapters/01-neuron/index.md#143)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**27.** 学习率越大，训练一定越快、效果一定越好。

</div>

??? tip "答案与解析"
    **答案：×**

    学习率太大会来回跳甚至发散。实验 2-3 中学习率 5.0 的准确率只有 11.4%。

    考查：学习率 · 认知层次：理解 · [回看 2.2.2](chapters/02-training/index.md#222)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**28.** 链式法则是把每一步的导数相加。

</div>

??? tip "答案与解析"
    **答案：×**

    是相乘，像齿轮传动一样，每一级的倍数乘起来。

    考查：链式法则 · 认知层次：记忆 · [回看 3.2.1](chapters/03-backprop/index.md#321)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**29.** 数据增强只对训练集做，验证集和测试集保持原样。

</div>

??? tip "答案与解析"
    **答案：√**

    验证和测试要反映模型在真实数据上的表现。

    考查：数据增强 · 认知层次：理解 · [回看 4.4.1](chapters/04-generalization/index.md#441)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**30.** 不管输入有几个通道，一个卷积核只输出一张特征图。

</div>

??? tip "答案与解析"
    **答案：√**

    卷积核各层与对应通道的乘加结果会加在一起，再加偏置，得到一张特征图。

    考查：通道 · 认知层次：理解 · [回看 5.2.3](chapters/05-cnn/index.md#523)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**31.** LSTM 的三道门的开关程度是工程师根据经验手工设定的。

</div>

??? tip "答案与解析"
    **答案：×**

    门的值由网络根据当前输入和隐藏状态计算，相关参数是从数据中学出来的。

    考查：LSTM · 认知层次：理解 · [回看 6.3](chapters/06-sequence/index.md#63)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**32.** 只要总准确率高，就说明模型在每一类上都表现很好。

</div>

??? tip "答案与解析"
    **答案：×**

    实验 7-3 中总准确率 91.44%，但链锯的精确率只有 0.781。要看混淆矩阵和每一类的指标。

    考查：混淆矩阵 · 认知层次：评价 · [回看 7.4.1](chapters/07-image-transfer/index.md#741)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="√" markdown>

**33.** `nn.Embedding` 中的词向量，开始是随机数，在训练中和其他参数一起学习。

</div>

??? tip "答案与解析"
    **答案：√**

    也可以用预先训练好的词向量（如 Word2Vec）作为初始值。

    考查：词嵌入 · 认知层次：理解 · [回看 8.1.3](chapters/08-text/index.md#813)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**34.** 降噪自编码器训练时，输入和目标都是加了噪声的图片。

</div>

??? tip "答案与解析"
    **答案：×**

    输入是加噪声的图片，目标是干净的原图，网络才能学会去噪。

    考查：降噪 · 认知层次：理解 · [回看 9.2.2](chapters/09-generative/index.md#922)
    { .quiz-meta }

<div class="quiz quiz-judge" data-answer="×" markdown>

**35.** 强化学习的得分曲线和分类任务的损失曲线一样，通常会平稳地一直变好。

</div>

??? tip "答案与解析"
    **答案：×**

    实验 10-1 中得分在上升后有明显起伏，强化学习的训练通常更不稳定。

    考查：DQN · 认知层次：理解 · [回看实验 10-1](chapters/10-drl-project/index.md#10-1)
    { .quiz-meta }

**四、简答题**（每题 10 分）

**36.** 为什么在训练集和测试集之外，还需要一个验证集？如果直接用测试集来调参，会有什么问题？

??? tip "答案与解析"
    **参考答案**：调参、选模型、决定训练轮数这些选择，本身也是根据数据在“学习”。如果每次都看测试集的成绩来做选择，最后挑出的设置就是“照着测试集挑的”，测试集的成绩会偏高，不能代表模型在真正的新数据上的表现，这是一种数据泄露。所以要另外分出验证集，所有选择都只看验证集，测试集等一切定下来后只考一次，才能得到可信的最终成绩。

    考查：验证集、数据泄露 · 认知层次：理解 · [回看 4.1.2](chapters/04-generalization/index.md#412)
    { .quiz-meta }

**37.** 用“会骑自行车学电动车”的比喻解释迁移学习，并说明冻结和微调的区别。

??? tip "答案与解析"
    **参考答案**：会骑自行车的人已经掌握了平衡，学电动车只需学加速、刹车，不用从头学起。预训练模型在大量图片上学会了看边缘、纹理、形状这些通用本领，迁移到新任务时，只需学会“这些特征组合起来是哪一类”。冻结：预训练部分的参数不动，只训练新换上的最后一层，速度快、不易过拟合，适合数据很少时。微调：在冻结训练的基础上，解冻部分或全部参数，用很小的学习率一起训练，让特征也适应新数据，效果通常更好，适合数据中等时。

    考查：迁移学习、冻结、微调 · 认知层次：理解 · [回看 7.3](chapters/07-image-transfer/index.md#73)
    { .quiz-meta }

**五、实操题**（每题 10 分）

**38.** 输入 3×224×224 的照片，先经过 `nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`，再经过 `nn.MaxPool2d(2)`。分别算出两层输出的形状和卷积层的参数个数，并用 5.3 节的动画或 PyTorch 核对。

??? tip "答案与解析"
    **参考答案**：卷积层边长 ⌊(224 + 2×3 − 7) ÷ 2⌋ + 1 = ⌊111.5⌋ + 1 = 112，输出 64×112×112；参数 64 × (3 × 7 × 7 + 1) = 64 × 148 = 9472。池化后 64×56×56。编者用 PyTorch 核对结果相同。ResNet 的第一层就是这样的卷积。

    考查：特征图尺寸、参数个数 · 认知层次：应用 · [回看 5.3.2](chapters/05-cnn/index.md#532)
    { .quiz-meta }
