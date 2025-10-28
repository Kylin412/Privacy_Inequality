import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Colors and markers, aligned with SW_stack style
COLOR_CS_BAR = "#6B9BD2"   # 浅蓝（与柱状图CS一致基调）
COLOR_FS_BAR = "#FFA500"    # 浅橘（与柱状图FS一致基调）
COLOR_BNE_SW = "darkgreen"  # BNE SW 虚线在SW_stack中使用的深绿色
COLOR_BNE_CS = "red"        # BNE CS 虚线在SW_stack中使用的红色

LINE_COLOR_CS = COLOR_CS_BAR
LINE_COLOR_FS = COLOR_FS_BAR
LINE_COLOR_SW = "#333333"   # 深灰，避免与上面两色和柱状色混淆

MARKER_CS = "o"
MARKER_FS = "s"
MARKER_SW = "^"


def plot_sw_ratio(data_csv: str, save_root: str, bne_csv_path: str = None) -> None:
    """
    Plot ratios (CS, FS, SW): Simulation / BNE vs firm_count (n) as line charts.
    - data_csv: path to data.csv
    - save_root: output root (Picture/SW/SW_ratio.png will be generated)
    - bne_csv_path: path to BNE/BNE.csv; if None, default to project BNE/BNE.csv
    """
    df = pd.read_csv(data_csv)
    # Parse firm_count from round_id (same as plot_sw)
    df["firm_count"] = df["round_id"].apply(lambda x: int(x.split("-")[0]))

    # Simulation data: aggregate mean by firm_count
    sim_summary = df.groupby("firm_count").agg(
        consumer_surplus=("consumer_surplus", "mean"),
        firm_surplus=("firm_surplus", "mean")
    ).reset_index()
    sim_summary["social_welfare"] = sim_summary["consumer_surplus"] + sim_summary["firm_surplus"]

    # Load BNE
    if bne_csv_path is None:
        bne_csv_path = os.path.join(os.path.dirname(__file__), "..", "BNE", "BNE.csv")
    if not os.path.exists(bne_csv_path):
        print(f"⚠️ BNE baseline not found: {bne_csv_path}, skip SW ratio plot")
        return

    bne_df = pd.read_csv(bne_csv_path)
    bne_summary = bne_df[["firm_count", "consumer_surplus", "firm_surplus"]].copy()
    bne_summary.rename(columns={
        "consumer_surplus": "bne_consumer_surplus",
        "firm_surplus": "bne_firm_surplus"
    }, inplace=True)
    bne_summary["bne_social_welfare"] = bne_summary["bne_consumer_surplus"] + bne_summary["bne_firm_surplus"]

    # Align and merge
    merged = pd.merge(sim_summary, bne_summary, on="firm_count", how="inner")
    if merged.empty:
        print("⚠️ SW ratio: merged is empty, firm_count may mismatch")
        return

    # Compute ratios (simulation / BNE)
    # Avoid division by zero: when BNE=0, set NaN
    def safe_div(a, b):
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(b == 0, np.nan, a / b)
        return r

    merged["ratio_cs"] = safe_div(merged["consumer_surplus"].values, merged["bne_consumer_surplus"].values)
    merged["ratio_fs"] = safe_div(merged["firm_surplus"].values, merged["bne_firm_surplus"].values)
    merged["ratio_sw"] = safe_div(merged["social_welfare"].values, merged["bne_social_welfare"].values)

    # Output directory
    save_folder = os.path.join(save_root, "Picture", "SW")
    os.makedirs(save_folder, exist_ok=True)

    # Draw lines (style consistent with SW_stack: no title/axes labels, keep legend)
    x = merged["firm_count"].values

    plt.figure(figsize=(10, 6))
    plt.plot(x, merged["ratio_cs"].values, color=LINE_COLOR_CS, marker=MARKER_CS, linestyle="-", label="Consumer Surplus Ratio")
    plt.plot(x, merged["ratio_fs"].values, color=LINE_COLOR_FS, marker=MARKER_FS, linestyle="-", label="Firm Surplus Ratio")
    plt.plot(x, merged["ratio_sw"].values, color=LINE_COLOR_SW, marker=MARKER_SW, linestyle="-", label="Social Welfare Ratio")

    # Data labels (consistent readability with SW_stack)
    for i, v in enumerate(merged["ratio_cs"].values):
        if not np.isnan(v):
            plt.text(x[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    for i, v in enumerate(merged["ratio_fs"].values):
        if not np.isnan(v):
            plt.text(x[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    for i, v in enumerate(merged["ratio_sw"].values):
        if not np.isnan(v):
            plt.text(x[i], v + 0.02, f"{v:.2f}", ha="center", fontsize=10)

    # x ticks step = 1
    plt.xticks(range(int(np.nanmin(x)), int(np.nanmax(x)) + 1, 1))
    plt.tick_params(axis='both', which='major', labelsize=12, width=1.5)

    # Grid, layout, legend
    plt.grid(True, alpha=0.3)
    # No title/axis labels, consistent with SW_stack
    plt.legend(fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(save_folder, "SW_ratio.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✅ Saved: {out_path}")
