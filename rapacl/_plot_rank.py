import re
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

LOG_PATH = "/root/workspace/RaPaCL/outputs/rapacl_foldfull/densenet/hvg50/ours/5011.log"
# LOG_PATH = "./log_rank_plot.log"  # 실제 로그 파일 경로로 수정
OUT_PATH = "stage1_rank_fold_mean.png"


fold_re = re.compile(r"\[INFO\] Start Fold (\d+)")
stage1_re = re.compile(r"\[Stage1:[^\]]+\]\[Epoch (\d+)\]")
rank_re = re.compile(r"rank_p=([0-9.]+)\s+rank_r=([0-9.]+)")


# data[split][metric][epoch] = [fold0_value, fold1_value, ...]
data = {
    "train": {"rank_p": defaultdict(list), "rank_r": defaultdict(list)},
    "val": {"rank_p": defaultdict(list), "rank_r": defaultdict(list)},
}

current_fold = None

with open(LOG_PATH, "r", encoding="utf-8") as f:
    for line in f:
        fold_match = fold_re.search(line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            continue

        if "[Stage1:" not in line:
            continue

        epoch_match = stage1_re.search(line)
        if not epoch_match:
            continue

        epoch = int(epoch_match.group(1))

        # train part / val part 분리
        if "| val_loss=" not in line:
            continue

        train_part, val_part = line.split("| val_loss=", 1)

        train_rank = rank_re.search(train_part)
        val_rank = rank_re.search(val_part)

        if train_rank:
            data["train"]["rank_p"][epoch].append(float(train_rank.group(1)))
            data["train"]["rank_r"][epoch].append(float(train_rank.group(2)))

        if val_rank:
            data["val"]["rank_p"][epoch].append(float(val_rank.group(1)))
            data["val"]["rank_r"][epoch].append(float(val_rank.group(2)))


epochs = sorted(data["train"]["rank_p"].keys())

train_rank_p = [np.mean(data["train"]["rank_p"][e]) for e in epochs]
val_rank_p = [np.mean(data["val"]["rank_p"][e]) for e in epochs]

train_rank_r = [np.mean(data["train"]["rank_r"][e]) for e in epochs]
val_rank_r = [np.mean(data["val"]["rank_r"][e]) for e in epochs]


plt.figure(figsize=(8, 5))

plt.plot(epochs, train_rank_p, color="blue", linewidth=2, label="Train rank_p")
plt.plot(epochs, val_rank_p, color="orange", linewidth=2, label="Val rank_p")

plt.plot(epochs, train_rank_r, color="blue", linestyle="--", linewidth=2, label="Train rank_r")
plt.plot(epochs, val_rank_r, color="orange", linestyle="--", linewidth=2, label="Val rank_r")

plt.xlabel("Epoch")
plt.ylabel("Effective Rank")
plt.title("Stage1 Effective Rank (Fold-wise Mean)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)

print(f"saved: {OUT_PATH}")