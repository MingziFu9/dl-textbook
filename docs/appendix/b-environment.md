---
title: 附录 B 实验环境、数据与算力说明
description: 实验所需的软硬件、PyTorch 与工具库安装、数据和预训练模型的下载与国内替代办法、待实测实验清单、中文字体设置与课时安排建议
---

# 附录 B 实验环境、数据与算力说明

## B.1 软硬件要求

本书的实验不需要显卡。编者在 **4 核 CPU、无显卡、8 GB 以上内存**的电脑上运行了除实验 8-3 以外的全部编程实验，最慢的实验 7-1 约 5 分钟。普通笔记本电脑（Windows、macOS、Linux 都可以）就能完成。

| 走法 | 适用实验 | 说明 |
|---|---|---|
| 浏览器 | 实验 1-1 和各章的交互动画 | 不用安装任何软件 |
| 本机 CPU | 实验 1-2～10-1（实验 8-3 除外） | 按 B.2 节安装环境 |
| 云算力平台 | 实验 8-3；第 10 章综合项目中数据较大的选题 | 选带 GPU 的 PyTorch 镜像，按小时计费，做完及时关机 |

国内常见的云算力平台有 AutoDL、阿里云 PAI、百度 AI Studio 等，都提供预装 PyTorch 的镜像，一般几元钱一小时；新用户往往有免费时长。选平台时以学校的安排为准。

## B.2 安装 PyTorch 和工具库

建议使用 Python 3.10 或更高版本（编者用 Python 3.13）。先装 PyTorch，再装其他库：

```bash
# 1. 没有 NVIDIA 显卡的电脑：安装 CPU 版 PyTorch（体积小很多）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. 其他工具库：在 labs/ 目录下运行
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果第 1 步下载很慢，也可以直接用清华镜像安装：`pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple`（在 Windows 和 macOS 上装到的就是 CPU 版；在 Linux 上可能装到较大的 GPU 版，同样能用）。有 NVIDIA 显卡的电脑，按 PyTorch 官网 pytorch.org 首页的选择器生成安装命令。

编者实测通过的版本（`labs/requirements.txt`）：

| 库 | 版本 | 用途 |
|---|---|---|
| torch | 2.15.0 | 全书 |
| torchvision | 0.29.2 | 读手写数字等数据集、图像变换、ResNet18 结构 |
| numpy | 2.5.3 | 数组计算 |
| matplotlib | 3.11.2 | 画曲线和图片 |
| jieba | 0.42.1 | 第 8 章中文分词 |
| gensim | 4.4.0 | 第 8 章词向量 |
| gymnasium | 1.4.0 | 第 10 章小车平衡环境 |
| transformers | （待实测） | 只有实验 8-3 需要 |

版本不必完全一样，PyTorch 2.0 以上一般都能运行本书程序；运行结果的最后一两位小数可能和书上略有不同。

安装后检查：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

能打印出版本号就说明装好了；第二个值是 `False` 表示没有可用的显卡，不影响本书实验。

## B.3 数据与预训练模型的下载

本书的数据都**不用手动准备**：程序第一次用到某个数据时，会自动下载到 `labs/data/` 目录，以后直接读取本地文件。各数据集的内容、规模、来源和许可见下表和 `labs/data/README.md`。

| 数据 | 规模 | 来源与许可 | 使用章节 |
|---|---|---|---|
| MNIST 手写数字 | 7 万张，约 11 MB | Yann LeCun 等，常见许可说明为 CC BY-SA 3.0 | 第 1、2、3、4、9 章 |
| Fashion-MNIST 服装图片 | 7 万张，约 30 MB | Zalando Research，MIT 许可 | 第 9 章 |
| CIFAR-10 彩色小图 | 6 万张，约 140 MB | Alex Krizhevsky 等，供研究和教学使用 | 第 5、7 章 |
| Imagenette 10 类照片（160 像素版） | 约 1.3 万张，约 100 MB | fast.ai（Apache 2.0），图片来自 ImageNet，须遵守其非商业使用条款 | 第 5、7 章 |
| 外卖评论 waimai_10k | 约 1.2 万条，约 1 MB | SophonPlus ChineseNlpCorpus，仅限研究和教学 | 第 6、8 章 |
| 墨尔本每日最低气温 | 3 650 天，约 70 KB | 澳大利亚气象局原始数据，Jason Brownlee 整理 | 第 6 章 |
| ResNet18 预训练权重 | 约 45 MB | timm（pytorch-image-models），Apache 2.0 | 第 7 章 |
| 中文预训练模型 hfl/rbt3 | 约 150 MB | 哈工大讯飞联合实验室，Apache 2.0 | 实验 8-3 |

**自动下载失败怎么办？** CIFAR-10、Imagenette、外卖评论、气温数据和 ResNet18 权重都放在 GitHub 上，个别网络环境下访问较慢。可以：

1. 多运行几次，或换一个时间段、换一个网络（例如手机热点）再试；
2. 按 `labs/data/README.md` 列出的地址在浏览器中下载，文件名不要改，放到 `labs/data/` 目录下，`.zip` 文件不用解压，程序会自动处理；
3. 请老师把下载好的 `labs/data/` 整个文件夹拷贝给全班。

**ResNet18 权重为什么不用 torchvision 自带的？** torchvision 的 `resnet18(weights=…)` 会从 PyTorch 官方服务器下载权重，国内部分网络访问困难。本书改用 timm 项目在 GitHub 上发布的同结构权重（在 ImageNet-1k 上训练，准确率与官方权重相当），用 `weights_only=True` 方式读取，只读参数、不执行文件中的代码。

**Hugging Face 模型（实验 8-3）**：国内可以使用镜像站，运行程序前先设置环境变量：

```bash
# Linux、macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows（命令提示符）
set HF_ENDPOINT=https://hf-mirror.com
```

也可以在 ModelScope（魔搭社区）搜索同名模型下载到本地，再把程序中的模型名改成本地文件夹路径。

## B.4 待实测的实验

下表列出编者的编写环境无法完整运行、书中没有给出运行结果的实验。请在能运行的环境中完成，把结果填入正文的记录表。

| 实验 | 需要什么 | 说明 |
|---|---|---|
| [实验 8-3](../chapters/08-text/index.md#8-3) 用中文预训练模型微调 | 能下载 Hugging Face 模型的网络（或镜像）；`pip install transformers`；建议有 GPU | 代码已按 transformers 库的标准用法编写，尚未完整运行。在 CPU 上也能跑，但比较慢；在云算力平台的 GPU 上约几分钟 |

其余实验的运行结果都是编者实际运行得到的。

## B.5 图片上的中文显示成方框

Matplotlib 默认不带中文字体。本书程序调用 `dlbook.py` 中的 `use_chinese_font()`，会依次尝试 SimHei（黑体）、Microsoft YaHei（微软雅黑）、PingFang SC（苹方）、WenQuanYi Zen Hei（文泉驿）、Noto Sans CJK SC。Windows 和 macOS 一般自带其中一种；Linux 上如果仍显示方框，安装一个中文字体，再删除 Matplotlib 的字体缓存：

```bash
sudo apt install fonts-noto-cjk          # Ubuntu、Debian
rm -rf ~/.cache/matplotlib               # 删除字体缓存，下次运行自动重建
```

## B.6 实验用时与课时安排建议

编者用 4 核 CPU 实测的程序运行时间（不含第一次下载数据的时间）：

| 章 | 实验与运行时间 |
|---|---|
| 第 1 章 | 1-1 浏览器；1-2 不到 10 秒 |
| 第 2 章 | 2-1 不到 1 秒；2-2 约 10 秒；2-3 约 1 分钟 |
| 第 3 章 | 3-1 不到 1 秒；3-2 约 10 秒 |
| 第 4 章 | 4-1 约 10 秒；4-2 约 25 秒；4-3 约 20 秒 |
| 第 5 章 | 5-1 几秒；5-2 约 1 分钟；5-3 约 2.5 分钟 |
| 第 6 章 | 每个都在 30 秒内 |
| 第 7 章 | 7-1 约 5 分钟；7-2 约 3 分钟（补充程序约 30 秒）；7-3 约 30 秒 |
| 第 8 章 | 8-1、8-2 各约 1 分钟；8-3 待实测 |
| 第 9 章 | 9-1 约 30 秒；9-2 约 2.5 分钟；9-3 约 1 分钟 |
| 第 10 章 | 10-1 约 1～2 分钟；10-2 为小组项目 |

课时安排建议：

- 本课程 18 个实验学时，每章 1～2 学时，每章的实验多于学时能做完的数量。建议每章至少做全部 ★ 基础实验，☆ 拓展实验由教师按学生情况选做或留作课后练习。
- 每个实验的“用时”（20～45 分钟）包括读步骤、运行、观察结果和填写记录表，程序本身的运行时间都不长。运行较慢的实验（5-3、7-1、7-2、9-2）可以先让程序跑起来，等待期间讲解代码。
- 第 10 章的综合项目（实验 10-2）安排在课余 2～3 周完成，最后用 2 个线下学时汇报互评。
- 每章有 1 个线下学时，建议用于答疑和重点讲解：第 1 章帮学生装好环境，第 4 章集中讲读曲线和调参，第 7 章带着学生做一遍迁移学习，第 10 章用于项目汇报。
