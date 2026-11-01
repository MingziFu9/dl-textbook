// 学习图谱数据：由编者整理，概念与定义取自附录 D 术语表
// 改完后运行 python tools/graph_text.py，检查数据并更新 index.md 里的文字版
var CATS = ["示例类别一", "示例类别二"];
var CHAPTERS = ["", "初识神经网络", "损失与梯度下降", "反向传播", "过拟合与调优", "卷积神经网络", "RNN 到 Transformer", "图像分类与迁移学习", "文本分类与预训练", "生成模型", "强化学习与综合项目"];
var CHDIRS = ["", "01-neuron", "02-training", "03-backprop", "04-generalization", "05-cnn", "06-sequence", "07-image-transfer", "08-text", "09-generative", "10-drl-project"];
var NODES = [
  {"id": "示例术语", "ch": 1, "cat": 0, "a": "g-示例术语", "d": "这里是术语表中的定义，鼠标悬停时显示"},
  {"id": "另一个术语", "ch": 1, "cat": 1, "a": "g-另一个术语", "d": "以示例术语为基础的概念"}
];
var EDGES = [["示例术语", "另一个术语"]];
