# 第 9 章 生成模型：让网络学会“创作”：实验代码

对应教材：[第 9 章](https://mingzifu9.github.io/dl-textbook/chapters/09-generative/)。所有程序都在 `labs/` 目录下运行，例如：

```bash
cd labs
python ch09/lab9_1_autoencoder.py
```

| 实验 | 类型 | 用时 | 内容 | 程序 |
|---|---|---|---|---|
| 9-1 | ★ 基础 | 40 分钟 | 自编码器压缩与还原手写数字；降噪自编码器去噪；用还原误差找出服装图片（异常检测） | `lab9_1_autoencoder.py` |
| 9-2 | ★ 基础 | 45 分钟 | 训练一个小 GAN 生成手写数字，观察生成效果随轮数的变化，读两条损失曲线 | `lab9_2_gan.py` |
| 9-3 | ☆ 拓展 | 30 分钟 | 二维点的扩散模型：正向加噪、训练去噪网络、从随机点生成“笑脸” | `lab9_3_diffusion_2d.py` |

程序运行时生成的图片（`.png`）和模型文件（`.pt`）保存在本目录，不提交到仓库。数据第一次使用时自动下载到 `labs/data/`，见 `labs/data/README.md`。
