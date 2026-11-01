"""实验 4-3 小网格调参，写一份调参记录

在实验 4-1 的数据上，把学习率（3 种）× 隐藏层大小（2 种）× Dropout（2 种）共 12 种组合各训练 30 轮，
只看验证集选出最好的一组，最后才用测试集考一次。
运行：python lab4_3_grid.py（在 labs/ch04 目录下，约 1 分钟）
"""
import itertools
import time

from common04 import X_TEST, Y_TEST, evaluate, fit, make_mlp, set_seed

records = []
start = time.perf_counter()
print("编号  学习率   隐藏层  Dropout  最高验证准确率（第几轮）")
for i, (lr, hidden, dropout) in enumerate(itertools.product([1e-4, 1e-3, 1e-2], [64, 512], [0.0, 0.5]), 1):
    set_seed(42)
    model = make_mlp(hidden=hidden, dropout=dropout)
    hist = fit(model, epochs=30, lr=lr)
    best_acc = max(hist["val_acc"])
    best_ep = hist["val_acc"].index(best_acc) + 1
    records.append((best_acc, i, lr, hidden, dropout, best_ep))
    print(f"{i:>3}   {lr:<7}{hidden:>6}{dropout:>8}      {best_acc:.3f}（第 {best_ep} 轮）")
print(f"12 组共用时 {time.perf_counter() - start:.0f} 秒")

best_acc, i, lr, hidden, dropout, best_ep = max(records)
print(f"\n验证集上最好的是第 {i} 组：学习率 {lr}，隐藏层 {hidden}，Dropout {dropout}，第 {best_ep} 轮验证准确率 {best_acc:.3f}")
set_seed(42)
model = make_mlp(hidden=hidden, dropout=dropout)
fit(model, epochs=best_ep, lr=lr)            # 按选出的设置重新训练到最好的那一轮
_, test_acc = evaluate(model, X_TEST, Y_TEST)
print(f"用这组设置重新训练 {best_ep} 轮，在测试集上考一次：准确率 {test_acc:.4f}")
