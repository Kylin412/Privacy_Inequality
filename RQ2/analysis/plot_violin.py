import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Load theoretical baseline ===
bne_path = os.path.join(os.path.dirname(__file__), "..", "BNE", "BNE.csv")
BNE_DF = pd.read_csv(bne_path) if os.path.exists(bne_path) else None


def plot_violin(root_exp_dir, save_root, n_consumers=20, shift_fix=True):
    """
    Plot violin charts for data under the consumer folder
    root_exp_dir: an experiment result folder, e.g., all_results/exp1
    save_root: output path (usually same as root_exp_dir)
    n_consumers: number of consumers in experiment (default 20)
    shift_fix: whether to manually adjust Seaborn/Matplotlib x-axis shift (shift lines by -1)
    """

    consumer_dir = os.path.join(root_exp_dir, "consumer")
    if not os.path.exists(consumer_dir):
        print(f"❌ Consumer folder not found: {consumer_dir}")
        return

    # === Collect consumer revenue data ===
    all_data = []
    for file in os.listdir(consumer_dir):
        if file.endswith(".csv") and file.startswith("firm"):
            try:
                firm_count = int(file.replace("firm", "").replace(".csv", ""))
            except:
                continue
            df = pd.read_csv(os.path.join(consumer_dir, file))
            if "total_revenue" not in df.columns:
                continue
            for v in df["total_revenue"].dropna().values:
                all_data.append([firm_count, v])

    data = pd.DataFrame(all_data, columns=["firm_count", "total_revenue"])
    if data.empty:
        print(f"⚠️ consumer data is empty: {consumer_dir}")
        return

    # === Ensure firm_count numeric, order ensures x-axis sorted ===
    data["firm_count"] = data["firm_count"].astype(int)
    order = sorted(data["firm_count"].unique())

    # === Draw violin plot ===
    plt.figure(figsize=(10, 6))
    sns.violinplot(
        x="firm_count", y="total_revenue",
        data=data, inner=None, color="lightgray", order=order
    )

    # === Median line ===
    medians = data.groupby("firm_count")["total_revenue"].median().reset_index()
    x_m = medians["firm_count"].values
    y_m = medians["total_revenue"].values
    if shift_fix:
        x_m = x_m - 1  # manual shift
    plt.plot(x_m, y_m, color="blue", marker="o", label="Median")

    # === Optional: add mean points (red triangles) ===
    means = data.groupby("firm_count")["total_revenue"].mean().reset_index()
    x_mean = means["firm_count"].values
    y_mean = means["total_revenue"].values
    if shift_fix:
        x_mean = x_mean - 1
    plt.scatter(x_mean, y_mean, color="red", marker="^", label="Mean")

    # === Theoretical baseline line (BNE consumer_surplus / n_consumers) ===
    if BNE_DF is not None:
        bne_x = BNE_DF["firm_count"].values
        bne_y = (BNE_DF["consumer_surplus"].values) / n_consumers
        if shift_fix:
            bne_x = bne_x - 1
        plt.plot(bne_x, bne_y, color="green", linestyle="--", marker="*", label=f"Theory (BNE/{n_consumers})")

    # === Beautify ===
    plt.title("Consumer Total Revenue Distribution by Firm Count")
    plt.xlabel("Firm Count")
    plt.ylabel("Consumer Total Revenue")
    plt.legend()
    plt.tight_layout()

    # === Save figure ===
    save_dir = os.path.join(save_root, "Picture", "Violin")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "consumer_violin.png"), dpi=300, bbox_inches="tight")
    plt.close()