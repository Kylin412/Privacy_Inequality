import os
import numpy as np
import pandas as pd
from scipy import stats
import itertools


def paired_ttest(x, y):
    """Paired t-test, return p-value"""
    _, p = stats.ttest_rel(x, y, nan_policy="omit")
    return p


def paired_wilcoxon(x, y):
    """Paired Wilcoxon test, return p-value"""
    try:
        _, p = stats.wilcoxon(x, y)
    except ValueError:  # 比如全 0 或配对样本长度不足会报错
        p = float("nan")
    return p


def compare_all_experiments(root="all_results", save_csv=True):
    """
    For all experiments under all_results, perform pairwise tests and output matrix CSV.
    :param root: root directory under all_results
    :param save_csv: whether to save CSV files
    """
    # Only select experiment directories with required files; exclude non-experiment dirs
    candidates = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    exps = []
    for d in candidates:
        if d in {"pairwise_tests", "reports"}:
            continue
        data_path = os.path.join(root, d, "data.csv")
        gini_path = os.path.join(root, d, "gini.csv")
        if os.path.exists(data_path) and os.path.exists(gini_path):
            exps.append(d)
    exps.sort()
    print(f"🧪 Experiments found: {exps}")

    results = {}

    # Metrics to compare
    metrics = {
        "consumer_surplus": "data",
        "firm_surplus": "data",
        "consumer_gini": "gini",
        "firm_gini": "gini",
    }

    # Initialize result matrices for each metric
    for metric in metrics:
        results[f"{metric}_ttest"] = pd.DataFrame(index=exps, columns=exps)
        results[f"{metric}_wilcoxon"] = pd.DataFrame(index=exps, columns=exps)

    # Iterate over all experiment pairs
    for exp1, exp2 in itertools.combinations(exps, 2):
        file1_data = os.path.join(root, exp1, "data.csv")
        file2_data = os.path.join(root, exp2, "data.csv")
        file1_gini = os.path.join(root, exp1, "gini.csv")
        file2_gini = os.path.join(root, exp2, "gini.csv")

        # Read (defensively: skip if missing)
        if not (os.path.exists(file1_data) and os.path.exists(file2_data) and os.path.exists(file1_gini) and os.path.exists(file2_gini)):
            continue
        data1 = pd.read_csv(file1_data)
        data2 = pd.read_csv(file2_data)
        gini1 = pd.read_csv(file1_gini)
        gini2 = pd.read_csv(file2_gini)

        # Merge data
        data_merge = pd.merge(
            data1[["round_id", "consumer_surplus", "firm_surplus"]],
            data2[["round_id", "consumer_surplus", "firm_surplus"]],
            on="round_id",
            suffixes=("_1", "_2"),
        )

        # Merge gini
        gini_merge = pd.merge(
            gini1[["round_id", "consumer_gini", "firm_gini"]],
            gini2[["round_id", "consumer_gini", "firm_gini"]],
            on="round_id",
            suffixes=("_1", "_2"),
        )

        # Surplus tests
        for m in ["consumer_surplus", "firm_surplus"]:
            x = data_merge[f"{m}_1"]
            y = data_merge[f"{m}_2"]
            p_t = paired_ttest(x, y)
            p_w = paired_wilcoxon(x, y)
            results[f"{m}_ttest"].loc[exp1, exp2] = results[f"{m}_ttest"].loc[exp2, exp1] = p_t
            results[f"{m}_wilcoxon"].loc[exp1, exp2] = results[f"{m}_wilcoxon"].loc[exp2, exp1] = p_w

        # Gini tests
        for m in ["consumer_gini", "firm_gini"]:
            x = gini_merge[f"{m}_1"]
            y = gini_merge[f"{m}_2"]
            p_t = paired_ttest(x, y)
            p_w = paired_wilcoxon(x, y)
            results[f"{m}_ttest"].loc[exp1, exp2] = results[f"{m}_ttest"].loc[exp2, exp1] = p_t
            results[f"{m}_wilcoxon"].loc[exp1, exp2] = results[f"{m}_wilcoxon"].loc[exp2, exp1] = p_w

    # Save results
    if save_csv:
        save_dir = os.path.join(root, "pairwise_tests")
        os.makedirs(save_dir, exist_ok=True)
        for k, df in results.items():
            path = os.path.join(save_dir, f"{k}.csv")
            df.to_csv(path)
            print(f"✅ Saved matrix: {path}")

    print("🎉 All tests completed!")
    return results


def _ttest_1samp_safe(x: np.ndarray, popmean: float):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size < 2 or np.isnan(popmean):
        return float("nan")
    try:
        _, p = stats.ttest_1samp(x, popmean, nan_policy="omit")
        return float(p)
    except Exception:
        return float("nan")


def _sign_test_against_value(x: np.ndarray, value: float):
    """Sign test: median equals value null hypothesis via binomial test.
    Ignore samples equal to value; count >value vs <value. Return two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0 or np.isnan(value):
        return float("nan")
    pos = np.sum(x > value)
    neg = np.sum(x < value)
    n = int(pos + neg)
    if n == 0:
        return float("nan")
    try:
        res = stats.binomtest(k=int(pos), n=n, p=0.5, alternative="two-sided")
        return float(res.pvalue)
    except Exception:
        return float("nan")


def _format_p(p):
    if np.isnan(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def compare_vs_bne_and_markdown(root: str = "all_results", bne_csv: str = os.path.join("BNE", "BNE.csv")):
    """
    For each experiment and firm_count, compare samples to BNE via:
    - One-sample t-test (mean equals BNE)
    - Sign test (median equals BNE)

    Metrics:
    - data.csv: consumer_surplus, firm_surplus, SW, WR, consumer_share
    - gini.csv: consumer_gini, firm_gini

    Output: Markdown tables under root/reports by metric×test, columns=firm_count, rows=experiment, cell=p-value.
    """
    # Read BNE
    if not os.path.exists(bne_csv):
        raise FileNotFoundError(f"BNE CSV not found: {bne_csv}")
    bne_df = pd.read_csv(bne_csv)
    # Normalize columns
    # Expected: firm_count, consumer_surplus, firm_surplus, social_welfare, CR, consumer_gini, firm_gini
    if "firm_count" not in bne_df.columns:
        raise ValueError("BNE.csv missing firm_count column")

    # Derived BNE column WR (if computable)
    if {"consumer_surplus", "firm_surplus"}.issubset(set(bne_df.columns)):
        with np.errstate(divide='ignore', invalid='ignore'):
            bne_df["WR"] = bne_df["consumer_surplus"] / bne_df["firm_surplus"]
    else:
        bne_df["WR"] = np.nan

    firm_counts = sorted(bne_df["firm_count"].dropna().unique().tolist())

    # Collect experiment directories
    exps = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    exps = [e for e in exps if e != "pairwise_tests"]
    exps.sort()

    reports_dir = os.path.join(root, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # Prepare results containers: {metric: {test: DataFrame}}
    metrics_tests = {
        "consumer_surplus": ["ttest", "sign"],
        "firm_surplus": ["ttest", "sign"],
        "SW": ["ttest", "sign"],
        "WR": ["ttest", "sign"],
        "consumer_share": ["ttest", "sign"],
        "consumer_gini": ["ttest", "sign"],
        "firm_gini": ["ttest", "sign"],
    }

    results = {m: {t: pd.DataFrame(index=exps, columns=firm_counts) for t in tests}
               for m, tests in metrics_tests.items()}

    # Iterate experiments, build samples and test against BNE
    for exp in exps:
        exp_dir = os.path.join(root, exp)
        data_path = os.path.join(exp_dir, "data.csv")
        gini_path = os.path.join(exp_dir, "gini.csv")

        # Read data.csv
        if os.path.exists(data_path):
            ddf = pd.read_csv(data_path)
            # Parse firm_count
            if "firm_count" not in ddf.columns:
                ddf["firm_count"] = ddf["round_id"].astype(str).apply(lambda x: int(str(x).split("-")[0]))
            # Derived metrics
            ddf["SW"] = ddf["consumer_surplus"] + ddf["firm_surplus"]
            with np.errstate(divide='ignore', invalid='ignore'):
                ddf["WR"] = ddf["consumer_surplus"] / ddf["firm_surplus"]
                ddf["consumer_share"] = ddf["consumer_surplus"] / (ddf["consumer_surplus"] + ddf["firm_surplus"]) 
            # Clean invalid
            for col in ["WR", "consumer_share", "SW"]:
                if col in ddf.columns:
                    ddf[col] = ddf[col].replace([np.inf, -np.inf], np.nan)
        else:
            ddf = None

        # Read gini.csv
        if os.path.exists(gini_path):
            gdf = pd.read_csv(gini_path)
            if "firm_count" not in gdf.columns:
                if "firm_num" in gdf.columns:
                    gdf = gdf.rename(columns={"firm_num": "firm_count"})
        else:
            gdf = None

        # For each firm_count perform tests
        for fc in firm_counts:
            bne_row = bne_df[bne_df["firm_count"] == fc]
            if bne_row.empty:
                continue
            bne_row = bne_row.iloc[0]

            # data metrics
            if ddf is not None:
                sub = ddf[ddf["firm_count"] == fc]
                # Mapping: metric -> BNE column name
                mapping = {
                    "consumer_surplus": "consumer_surplus",
                    "firm_surplus": "firm_surplus",
                    "SW": "social_welfare",
                    "WR": "WR",
                    "consumer_share": "CR",
                }
                for metric, bne_col in mapping.items():
                    if metric not in sub.columns or bne_col not in bne_row:
                        continue
                    x = sub[metric].to_numpy(dtype=float)
                    mu = float(bne_row[bne_col]) if bne_col in bne_row.index else float("nan")
                    p_t = _ttest_1samp_safe(x, mu)
                    p_s = _sign_test_against_value(x, mu)
                    results[metric]["ttest"].loc[exp, fc] = _format_p(p_t)
                    results[metric]["sign"].loc[exp, fc] = _format_p(p_s)

            # gini metrics
            if gdf is not None:
                gsub = gdf[gdf["firm_count"] == fc]
                for metric, bne_col in [("consumer_gini", "consumer_gini"), ("firm_gini", "firm_gini")]:
                    if metric not in gsub.columns or bne_col not in bne_row:
                        continue
                    x = gsub[metric].to_numpy(dtype=float)
                    mu = float(bne_row[bne_col]) if bne_col in bne_row.index else float("nan")
                    p_t = _ttest_1samp_safe(x, mu)
                    p_s = _sign_test_against_value(x, mu)
                    results[metric]["ttest"].loc[exp, fc] = _format_p(p_t)
                    results[metric]["sign"].loc[exp, fc] = _format_p(p_s)

    # Write Markdown reports
    for metric, tests in metrics_tests.items():
        for test in tests:
            df = results[metric][test]
            df = df.reindex(index=exps, columns=firm_counts)
            md_lines = []
            md_lines.append(f"# {metric} vs BNE - {('One-sample t-test' if test=='ttest' else 'Sign test')}\n")
            # Build Markdown table
            header = "| Experiment | " + " | ".join(str(fc) for fc in firm_counts) + " |"
            sep = "|---" * (len(firm_counts) + 1) + "|"
            md_lines.append(header)
            md_lines.append(sep)
            for exp in df.index:
                row_vals = [str(df.loc[exp, fc]) if fc in df.columns else "" for fc in firm_counts]
                md_lines.append("| " + exp + " | " + " | ".join(row_vals) + " |")
            path = os.path.join(reports_dir, f"{metric}_{test}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))
            print(f"✅ Generated report: {path}")

    return results


if __name__ == "__main__":
    compare_all_experiments("all_results")
    # Extra: one-sample and sign test vs BNE
    compare_vs_bne_and_markdown("all_results", os.path.join("..", "BNE", "BNE.csv"))