# 实验数据说明

本目录存放实验用到的数据和预训练模型。它们都**不提交到仓库**：第一次运行用到它的程序时，`dlbook.py` 或 torchvision 会自动下载到这里，以后直接读取。程序还会在这里生成 `*.pt` 缓存文件（把图片缩放后存起来，第二次运行更快），可以随时删除。

## 文件清单

| 文件或文件夹 | 内容 | 规模 | 来源与许可 | 使用章节 |
|---|---|---|---|---|
| `MNIST/` | 手写数字 0～9，28×28 灰度图 | 训练 6 万张、测试 1 万张，约 11 MB | Yann LeCun 等，torchvision 自动下载；常见许可说明为 CC BY-SA 3.0 | 第 1、2、3、4、9 章 |
| `FashionMNIST/` | 10 类服装图片，28×28 灰度图 | 7 万张，约 30 MB | Zalando Research，MIT 许可，torchvision 自动下载 | 第 9 章（异常检测） |
| `cifar10.zip`、`cifar10/` | 10 类彩色小图（飞机、汽车、鸟……），32×32 | 训练 5 万张、测试 1 万张，约 140 MB | Alex Krizhevsky 等（多伦多大学），供研究和教学使用；下载的是 Ultralytics 整理的图片文件夹版本 | 第 5、7 章 |
| `imagenette160.zip`、`imagenette160/` | ImageNet 中 10 个容易区分的类别（丁鲷鱼、史宾格犬、收音机、链锯、教堂、圆号、垃圾车、加油泵、高尔夫球、降落伞），短边 160 像素 | 训练约 9 470 张、验证 3 925 张，约 100 MB | fast.ai 的 Imagenette 项目（Apache 2.0），图片本身来自 ImageNet，须遵守 ImageNet 的非商业使用条款；下载的是 Ultralytics 整理的版本 | 第 5、7 章 |
| `waimai_10k.csv` | 外卖平台用户评论，标签 1 为好评、0 为差评 | 约 1.2 万条（好评约 4 000、差评约 8 000），约 1 MB | SophonPlus 的 ChineseNlpCorpus 项目整理，仅限研究和教学使用 | 第 6、8 章 |
| `daily-min-temperatures.csv` | 澳大利亚墨尔本 1981～1990 年每日最低气温 | 3 650 天，约 70 KB | 澳大利亚气象局原始数据，经 Jason Brownlee 的 Datasets 项目整理 | 第 6 章 |
| `resnet18_a1_0-d63eafa0.pth` | 在 ImageNet-1k 上训练好的 ResNet18 权重 | 约 45 MB | timm（pytorch-image-models）项目发布，Apache 2.0 许可 | 第 7 章 |

第 10 章的 CartPole 小车平衡环境由 gymnasium 库自带，不用下载。实验 8-3（待实测）用到的 `hfl/rbt3` 模型（哈工大讯飞联合实验室，Apache 2.0，约 150 MB）由 transformers 库下载到它自己的缓存目录，不放在这里。

## 下载地址（自动下载失败时手动下载）

| 文件 | 地址 |
|---|---|
| `cifar10.zip` | https://github.com/ultralytics/assets/releases/download/v0.0.0/cifar10.zip |
| `imagenette160.zip` | https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenette160.zip |
| `waimai_10k.csv` | https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv |
| `daily-min-temperatures.csv` | https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv |
| `resnet18_a1_0-d63eafa0.pth` | https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth |

把下载好的文件直接放在本目录下（文件名不要改），`.zip` 文件不用自己解压，程序会自动解压。MNIST 和 Fashion-MNIST 由 torchvision 下载，失败时可以多试几次，或从同学、老师那里复制整个 `MNIST/`、`FashionMNIST/` 文件夹过来。

**使用说明**：公开数据集仅用于教学和研究，请遵守原项目的许可说明，不得用于商业用途。
