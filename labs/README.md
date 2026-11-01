# 实验代码

《神经网络与深度学习》的配套实验代码，只用 PyTorch 一套代码。按章存放：`chNN/` 对应第 NN 章，文件名为 `labN_M_英文短名.py`（第 N 章实验 N-M），每章的 `README.md` 列出了实验、用时和程序。`dlbook.py` 是各章共用的小工具（下载并读取数据、画训练曲线、设置中文字体和随机种子），公共数据放在 `data/`，见 `data/README.md`。

## 安装实验环境

编者在 Python 3.13、4 核 CPU、无显卡的 Linux 电脑上运行了全部程序（实验 8-3 除外），Python 3.10 及以上版本都可以。没有显卡的电脑先装 CPU 版 PyTorch，体积小很多：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

有 NVIDIA 显卡的电脑按 PyTorch 官网（pytorch.org）的说明选择对应的 CUDA 版本安装。

## 运行

每个程序都在 `labs/` 目录下运行：

```bash
cd labs
python ch02/lab2_2_mnist_mlp.py
```

数据和预训练模型第一次使用时自动下载到 `data/`，以后直接读取。下载失败时，按 `data/README.md` 中的地址手动下载。程序都固定了随机种子，同一台电脑上结果可以重复；换电脑、换版本后最后一两位小数可能不同。

## 运行条件特殊的实验

| 实验 | 说明 |
|---|---|
| 实验 1-1 | 只用浏览器，在教材网页的“网络游乐场”动画中完成 |
| 实验 8-3 | 待实测：需要另装 `transformers`，下载约 150 MB 的中文预训练模型 `hfl/rbt3`，建议在云算力平台上运行 |
| 实验 10-2 | 小组项目，没有统一程序 |
