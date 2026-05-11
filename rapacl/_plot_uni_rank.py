import re
import pandas as pd
import matplotlib.pyplot as plt

log_path = "/root/workspace/RaPaCL/outputs/rapacl_foldfull/densenet/hvg50/ours/5011.log"  ### 

pattern_fold = re.compile(r"\[INFO\] Start Fold (\d+)")
pattern_stage1 = re.compile(
    r"\[Stage1:(?P<stage>[^\]]+)\]\[Epoch (?P<epoch>\d+)\].*?"
    r"\|\s*acc=.*?"
    r"uni_p=(?P<uni_p>-?\d+\.\d+)\s+"
    r"uni_r=(?P<uni_r>-?\d+\.\d+)\s+"
    r"rank_p=(?P<rank_p>-?\d+\.\d+)\s+"
    r"rank_r=(?P<rank_r>-?\d+\.\d+)"
    r".*?\|\s*val_loss=.*?"
    r"uni_p=(?P<val_uni_p>-?\d+\.\d+)\s+"
    r"uni_r=(?P<val_uni_r>-?\d+\.\d+)\s+"
    r"rank_p=(?P<val_rank_p>-?\d+\.\d+)\s+"
    r"rank_r=(?P<val_rank_r>-?\d+\.\d+)"
)

records = []
current_fold = None

with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        m_fold = pattern_fold.search(line)
        if m_fold:
            current_fold = int(m_fold.group(1))
            continue

        m = pattern_stage1.search(line)
        if m and current_fold is not None:
            d = m.groupdict()
            records.append({
                "fold": current_fold,
                "epoch": int(d["epoch"]),
                "stage": d["stage"],

                "train_uni_p": float(d["uni_p"]),
                "train_uni_r": float(d["uni_r"]),
                "train_rank_p": float(d["rank_p"]),
                "train_rank_r": float(d["rank_r"]),

                "val_uni_p": float(d["val_uni_p"]),
                "val_uni_r": float(d["val_uni_r"]),
                "val_rank_p": float(d["val_rank_p"]),
                "val_rank_r": float(d["val_rank_r"]),
            })

df = pd.DataFrame(records)

mean_df = (
    df.groupby("epoch", as_index=False)
      .mean(numeric_only=True)
)

# -------------------------
# 1) Effective Rank plot
# -------------------------
plt.figure(figsize=(10, 6))

plt.plot(mean_df["epoch"], mean_df["train_rank_p"], label="Train rank_p", color="blue")
plt.plot(mean_df["epoch"], mean_df["val_rank_p"], label="Val rank_p", color="orange")

plt.plot(mean_df["epoch"], mean_df["train_rank_r"], label="Train rank_r", color="blue", linestyle="--")
plt.plot(mean_df["epoch"], mean_df["val_rank_r"], label="Val rank_r", color="orange", linestyle="--")

plt.title("Stage1 Effective Rank (Fold-wise Mean)")
plt.xlabel("Epoch")
plt.ylabel("Effective Rank")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("stage1_effective_rank_foldwise_mean.png", dpi=300)
plt.show()


# -------------------------
# 2) Uniformity plot
# -------------------------
plt.figure(figsize=(10, 6))

plt.plot(mean_df["epoch"], mean_df["train_uni_p"], label="Train uni_p", color="blue")
plt.plot(mean_df["epoch"], mean_df["val_uni_p"], label="Val uni_p", color="orange")

plt.plot(mean_df["epoch"], mean_df["train_uni_r"], label="Train uni_r", color="blue", linestyle="--")
plt.plot(mean_df["epoch"], mean_df["val_uni_r"], label="Val uni_r", color="orange", linestyle="--")

plt.title("Stage1 Uniformity (Fold-wise Mean)")
plt.xlabel("Epoch")
plt.ylabel("Uniformity")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("stage1_uniformity_foldwise_mean.png", dpi=300)
plt.show()