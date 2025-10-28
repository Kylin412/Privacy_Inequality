import os
import pandas as pd


def generate_summary(consumer_csv, firm_csv, out_dir):
    cdf = pd.read_csv(consumer_csv)
    fdf = pd.read_csv(firm_csv)

    cons = cdf.groupby(["firm_num", "round"])["total_revenue"].sum().reset_index(name="consumer_surplus")
    firms = fdf.groupby(["firm_num", "round"])["revenue"].sum().reset_index(name="firm_surplus")

    summary = cons.merge(firms, on=["firm_num", "round"], how="outer")
    summary["round_id"] = summary["firm_num"].astype(str) + "-" + summary["round"].astype(str)
    summary.rename(columns={"round": "Round"}, inplace=True)
    summary = summary[["round_id", "Round", "consumer_surplus", "firm_surplus"]]

    path = os.path.join(out_dir, "data.csv")
    summary.to_csv(path, index=False)
    return path