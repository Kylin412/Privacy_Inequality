import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import mean_ci

# Load theoretical baseline
bne_path = os.path.join(os.path.dirname(__file__), "..", "BNE", "BNE.csv")
BNE_DF = pd.read_csv(bne_path) if os.path.exists(bne_path) else None


def plot_sw(data_csv, save_root, confidence_list=[0.68, 0.90]):
    df = pd.read_csv(data_csv)
    df["firm_count"] = df["round_id"].apply(lambda x: int(x.split("-")[0]))
    df["SW"] = df["consumer_surplus"] + df["firm_surplus"]

    save_folder = os.path.join(save_root, "Picture", "SW")
    os.makedirs(save_folder, exist_ok=True)

    # ====== Aggregate means by firm_count ======
    summary = df.groupby("firm_count").agg(
        {"consumer_surplus": "mean", "firm_surplus": "mean"}).reset_index()
    summary["SW"] = summary["consumer_surplus"] + summary["firm_surplus"]

    # === Stacked bars ===
    plt.figure(figsize=(10, 6))
    b1 = plt.bar(summary["firm_count"], summary["consumer_surplus"],
                 label="Experimental Consumer Surplus", color="#6B9BD2")  # 更浅蓝色
    b2 = plt.bar(summary["firm_count"], summary["firm_surplus"],
                 bottom=summary["consumer_surplus"], label="Experimental Firm Surplus", color="#FFA500")  # 稍浅橘色

    # Numeric labels on bars
    for i, rect in enumerate(b1):
        if rect.get_height() > 0:
            plt.text(rect.get_x() + rect.get_width()/2, rect.get_height()/2,
                     f"{rect.get_height():.2f}", ha="center", fontsize=12)
    for i, rect in enumerate(b2):
        if rect.get_height() > 0:
            base = summary.loc[i, "consumer_surplus"]
            plt.text(rect.get_x()+rect.get_width()/2, base+rect.get_height()/2,
                     f"{rect.get_height():.2f}", ha="center", fontsize=12)
    for i, v in enumerate(summary["SW"]):
        plt.text(summary["firm_count"].iloc[i], v+0.02, f"{v:.2f}", ha="center", fontsize=12)
    
    # Set x ticks step to 1
    plt.xticks(range(int(summary["firm_count"].min()), int(summary["firm_count"].max()) + 1, 1))
    plt.tick_params(axis='both', which='major', labelsize=12, width=1.5)

    # === Keep only BNE baselines ===
    if BNE_DF is not None:
        # Compute BNE social welfare
        bne_social_welfare = BNE_DF["consumer_surplus"] + BNE_DF["firm_surplus"]
        plt.plot(BNE_DF["firm_count"], bne_social_welfare,
                 color="darkgreen", linestyle="--", marker="*", label="BNE Social Welfare")
        plt.plot(BNE_DF["firm_count"], BNE_DF["consumer_surplus"],
                 color="red", linestyle="--", marker="s", label="BNE Consumer Surplus")

    #plt.title("Average Total Social Welfare (Stacked with Lines)")
    #plt.xlabel("Firm Count"); plt.ylabel("Welfare")
    plt.legend(fontsize=12); plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "SW_stack.png"), dpi=300, bbox_inches="tight"); plt.close()

    # === Original line + confidence interval section ===
    metrics = {"firm_surplus": ("Firm Surplus", "FS"),
               "consumer_surplus": ("Consumer Surplus", "CS"),
               "SW": ("Total Social Welfare", "SW")}
    for m, (title, shortname) in metrics.items():
        x, means = [], []
        for fc, d in df.groupby("firm_count"):
            mean, _ = mean_ci(d[m].values, 0.95)
            means.append(mean); x.append(fc)
        x, means = np.array(x), np.array(means)

        plt.figure(figsize=(8, 5))
        plt.plot(x, means, marker="o", color="#4A7BA7", label="Simulated Mean")

        # BNE baseline line
        if BNE_DF is not None:
            bne_x = BNE_DF["firm_count"]
            col = "social_welfare" if m == "SW" else m
            bne_y = BNE_DF[col]
            plt.plot(bne_x, bne_y, linestyle="--", marker="*", color="red", label="BNE")

        # Confidence intervals
        for i, conf in enumerate(sorted(confidence_list, reverse=True)):
            ci = []
            for fc, d in df.groupby("firm_count"):
                _, h = mean_ci(d[m].values, conf); ci.append(h)
            ci = np.array(ci)
            plt.fill_between(x, means-ci, means+ci, alpha=0.2+i*0.15, label=f"{int(conf*100)}% CI")

        # Data labels
        for i, v in enumerate(means):
            if not np.isnan(v):
                plt.text(x[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

        #plt.title(f"{title} with Theory and CIs")
        #plt.xlabel("Firm Count"); plt.ylabel(title)
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(save_folder, f"{shortname}.png"), dpi=300, bbox_inches="tight"); plt.close()