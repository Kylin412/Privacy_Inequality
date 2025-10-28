import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import mean_ci

# Directly read BNE.csv
bne_path = os.path.join(os.path.dirname(__file__), "..", "BNE", "BNE.csv")
BNE_DF = pd.read_csv(bne_path) if os.path.exists(bne_path) else None


def plot_wr_cs(data_csv, save_root, confidence_list=[0.68, 0.90]):
    df = pd.read_csv(data_csv)
    df["firm_count"] = df["round_id"].apply(lambda x: int(x.split("-")[0]))
    df["SW"] = df["consumer_surplus"] + df["firm_surplus"]
    df["WR"] = df["consumer_surplus"] / df["firm_surplus"]
    df["consumer_share"] = df["consumer_surplus"] / df["SW"]

    metrics = {"WR": ("Welfare Ratio", "WR"), "consumer_share": ("Consumer Surplus Share", "CS")}
    save_folder = os.path.join(save_root, "Picture", "WR")
    os.makedirs(save_folder, exist_ok=True)

    for m, (title, shortname) in metrics.items():
        x, means = [], []
        for fc, d in df.groupby("firm_count"):
            mean, _ = mean_ci(d[m].values, 0.95)
            means.append(mean)
            x.append(fc)
        x, means = np.array(x), np.array(means) * 100

        plt.figure(figsize=(8, 5))
        plt.plot(x, means, marker="o", color="blue", label="Simulated Mean")

        # === Theoretical baseline line ===
        if BNE_DF is not None:
            if m == "WR":
                bne_x = BNE_DF["firm_count"]
                bne_y = (BNE_DF["consumer_surplus"] / BNE_DF["firm_surplus"]) * 100
            elif m == "consumer_share":
                bne_x = BNE_DF["firm_count"]
                bne_y = BNE_DF["CR"]   # 直接使用 CR 列
            plt.plot(bne_x, bne_y, linestyle="--", marker="*", color="green", label="Theory (BNE)")

        # === Confidence intervals ===
        for i, conf in enumerate(sorted(confidence_list, reverse=True)):
            ci = []
            for fc, d in df.groupby("firm_count"):
                _, h = mean_ci(d[m].values, conf)
                ci.append(h * 100)
            ci = np.array(ci)
            plt.fill_between(x, means - ci, means + ci, alpha=0.2 + i * 0.15, label=f"{int(conf*100)}% CI")

        # === Data labels ===
        for i, v in enumerate(means):
            if not np.isnan(v):
                plt.text(x[i], v + 1, f"{v:.1f}%", ha="center", fontsize=8)

        # Baselines
        if m == "WR":
            plt.axhline(100, color="gray", linestyle="--")
        elif m == "consumer_share":
            plt.axhline(50, color="gray", linestyle="--")

        plt.title(f"{title} with Theory and CIs")
        plt.xlabel("Firm Count"); plt.ylabel(title + " (%)")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{shortname}.png"), dpi=300, bbox_inches="tight")
        plt.close()