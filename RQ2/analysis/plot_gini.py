import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# BNE baseline CSV path
bne_path = os.path.join(os.path.dirname(__file__), "..", "BNE", "BNE.csv")
BNE_DF = pd.read_csv(bne_path) if os.path.exists(bne_path) else None


def plot_gini(consumer_csv, firm_csv, save_root):
    cdf = pd.read_csv(consumer_csv)
    fdf = pd.read_csv(firm_csv)

    # === Revised gini_formula (sorted + strictly clipped to [0,1]) ===
    def gini_formula(array):
        arr = np.array(array, dtype=float)
        arr = arr[arr >= 0]  # filter negatives (avoid out-of-range)
        n = len(arr)
        if n == 0:
            return np.nan
        arr = np.sort(arr)  # sort
        sumx = arr.sum()
        if sumx == 0:
            return 0.0
        # compute by the standard formula
        gini = (2 * np.sum((np.arange(1, n + 1)) * arr)) / (n * sumx) - (n + 1) / n
        # clip to [0,1]
        return max(0.0, min(1.0, gini))

    cons, firms = [], []
    for (fc, rnd), g in cdf.groupby(["firm_num", "round"]):
        cons.append([f"{fc}-{rnd}", rnd, fc, gini_formula(g["total_revenue"].values)])
    cdf_g = pd.DataFrame(cons, columns=["round_id", "Round", "firm_num", "consumer_gini"])

    for (fc, rnd), g in fdf.groupby(["firm_num", "round"]):
        firms.append([f"{fc}-{rnd}", rnd, fc, gini_formula(g["revenue"].values)])
    fdf_g = pd.DataFrame(firms, columns=["round_id", "Round", "firm_num", "firm_gini"])

    gini_summary = cdf_g.merge(fdf_g, on=["round_id", "Round", "firm_num"], how="outer")
    gini_summary = gini_summary.sort_values(by=["firm_num", "Round"]).reset_index(drop=True)
    gini_summary.to_csv(os.path.join(save_root, "gini.csv"), index=False)

    avg = gini_summary.groupby("firm_num")[["consumer_gini", "firm_gini"]].mean().reset_index()

    plt.figure(figsize=(8, 6))
    plt.plot(avg["firm_num"], avg["consumer_gini"], marker="o", color="blue", label="Consumer Gini (avg)")
    plt.plot(avg["firm_num"], avg["firm_gini"], marker="s", linestyle="--", color="red", label="Firm Gini (avg)")

    # Theoretical baseline
    if BNE_DF is not None:
        plt.plot(BNE_DF["firm_count"], BNE_DF["consumer_gini"],
                 linestyle="--", marker="*", color="green", label="Theory Consumer Gini")
        plt.plot(BNE_DF["firm_count"], BNE_DF["firm_gini"],
                 linestyle="--", marker="*", color="orange", label="Theory Firm Gini")

    # Data labels
    for i, v in enumerate(avg["consumer_gini"]):
        plt.text(avg["firm_num"].iloc[i], v + 0.005, f"{v:.3f}", color="blue", ha="center", fontsize=7)
    for i, v in enumerate(avg["firm_gini"]):
        plt.text(avg["firm_num"].iloc[i], v + 0.005, f"{v:.3f}", color="red", ha="center", fontsize=7)

    plt.title("Average Gini Coefficient vs Firm Count")
    plt.xlabel("Firm Count"); plt.ylabel("Average Gini Coefficient")
    plt.legend(); plt.tight_layout()
    save_dir = os.path.join(save_root, "Picture", "Gini")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "gini_trend.png"), dpi=300, bbox_inches="tight")
    plt.close()