"""本书实验的公共小工具：下载并读取数据、画训练曲线、设置中文字体和随机种子。

各章实验用法（实验文件开头已经写好，不用改）：
    import sys; sys.path.append("..")
    from dlbook import load_mnist, set_seed, ...

数据第一次使用时自动下载到 labs/data/ 下，之后直接读取本地文件。
下载地址都是国内一般能访问的公开地址；如果下载失败，按 labs/data/README.md 手动下载后放到对应文件夹。
"""
import csv
import logging
import random
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).resolve().parent / "data"
DATA.mkdir(exist_ok=True)

URLS = {
    "cifar10": "https://github.com/ultralytics/assets/releases/download/v0.0.0/cifar10.zip",
    "imagenette160": "https://github.com/ultralytics/assets/releases/download/v0.0.0/imagenette160.zip",
    "waimai": "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv",
    "temperature": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
}

CIFAR10_CN = ["飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车"]
IMAGENETTE_CN = {"n01440764": "丁鲷鱼", "n02102040": "史宾格犬", "n02979186": "收音机", "n03000684": "链锯",
                 "n03028079": "教堂", "n03394916": "圆号", "n03417042": "垃圾车", "n03425413": "加油泵",
                 "n03445777": "高尔夫球", "n03888257": "降落伞"}


def set_seed(seed=42):
    """固定随机种子，让每次运行的结果都一样（同一台电脑、同一版本）。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def use_chinese_font():
    """让 Matplotlib 画的图能显示中文。"""
    import matplotlib.pyplot as plt
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC", "WenQuanYi Zen Hei",
                                       "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False


def download(name, filename=None):
    """把 URLS[name] 下载到 labs/data/，已经下载过就跳过。"""
    path = DATA / (filename or Path(URLS[name]).name)
    if not path.exists():
        print(f"正在下载 {path.name}（只在第一次运行时下载）……")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def load_mnist(train=True):
    """MNIST 手写数字：训练集 6 万张、测试集 1 万张，28×28 灰度图。返回 (图片张量 N×1×28×28，取值 0～1；标签张量)。"""
    from torchvision import datasets
    ds = datasets.MNIST(DATA, train=train, download=True)
    x = ds.data.float().unsqueeze(1) / 255.0
    return x, ds.targets.clone()


def _load_folder(root, classes, n_per_class, size, cache):
    """把“一个类别一个文件夹”的图片读进内存，缩放到 size×size，缓存成 .pt 文件。"""
    if cache.exists():
        return torch.load(cache)
    from PIL import Image
    xs, ys = [], []
    for label, cls in enumerate(classes):
        files = sorted((root / cls).iterdir())[:n_per_class]
        for f in files:
            img = Image.open(f).convert("RGB")
            if img.size != (size, size):
                w, h = img.size                       # 先裁成正方形（取中间），再缩放
                s = min(w, h)
                img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
                img = img.resize((size, size), Image.BILINEAR)
            xs.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1))
            ys.append(label)
    data = (torch.stack(xs), torch.tensor(ys))       # 图片以 0～255 的整数保存，省内存
    torch.save(data, cache)
    return data


def load_cifar10(train=True, n_per_class=None):
    """CIFAR-10 彩色小图：10 类，32×32。n_per_class 指定每类取几张（训练集最多 5000，测试集最多 1000）。
    返回 (图片张量 N×3×32×32，取值 0～1；标签张量)。"""
    root = DATA / "cifar10"
    if not root.exists():
        with zipfile.ZipFile(download("cifar10")) as z:
            z.extractall(DATA)
    split = "train" if train else "test"
    n = n_per_class or (5000 if train else 1000)
    classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    x, y = _load_folder(root / split, classes, n, 32, DATA / f"cifar10_{split}_{n}.pt")
    return x.float() / 255.0, y


def load_imagenette(train=True, n_per_class=None, size=96):
    """Imagenette：从 ImageNet 里挑出的 10 类照片（160 像素版），缩放到 size×size。
    返回 (图片张量 N×3×size×size，取值 0～1；标签张量；中文类名列表)。"""
    root = DATA / "imagenette160"
    if not root.exists():
        with zipfile.ZipFile(download("imagenette160")) as z:
            z.extractall(DATA)
    split = "train" if train else "val"
    classes = sorted(IMAGENETTE_CN)
    n = n_per_class or 10000
    x, y = _load_folder(root / split, classes, n, size, DATA / f"imagenette_{split}_{n}_{size}.pt")
    return x.float() / 255.0, y, [IMAGENETTE_CN[c] for c in classes]


def load_waimai():
    """外卖评论（waimai_10k）：约 1.2 万条，label 1 为好评、0 为差评。返回 (评论列表, 标签列表)。"""
    path = download("waimai")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["review"] for r in rows], [int(r["label"]) for r in rows]


def load_temperature():
    """墨尔本 1981—1990 年每日最低气温（摄氏度），共 3650 天。返回 (日期列表, 气温 NumPy 数组)。"""
    path = download("temperature")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))[1:]
    return [r[0] for r in rows], np.array([float(r[1]) for r in rows], dtype="float32")


def plot_curves(history, path, title="训练曲线"):
    """history 形如 {"训练损失": [...], "验证损失": [...]}，画成折线图保存到 path。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    use_chinese_font()
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=120)
    for name, values in history.items():
        ax.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=name)
    ax.set_xlabel("轮次")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"曲线已保存到 {path}")
