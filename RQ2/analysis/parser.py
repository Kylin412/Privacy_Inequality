import os, re
import pandas as pd
from .utils import safe_eval_mem


def parse_log_to_csv(log_file, out_dir):
    consumer_temp, firm_temp = [], []
    current_firm, current_round = None, None

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # Match firm number
            m = re.search(r"firm_num=(\d+)", line)
            if m:
                current_firm = int(m.group(1))
                continue

            # Match round
            m = re.search(r"--- Round (\d+)/\d+ ---", line)
            if m:
                current_round = int(m.group(1))
                continue

            # Consumer temp_mem
            if "Consumer" in line and "temp_mem:" in line:
                mem_match = re.search(r"Consumer \d+ temp_mem: ({.*)", line)  # relaxed: no enforced closing }
                if mem_match:
                    mem = safe_eval_mem(mem_match.group(1))
                    consumer_temp.append({
                        "firm_num": mem.get("num_firms", current_firm),
                        "round": current_round,
                        "consumer_id": f"C{mem.get('index', -1)+1}",
                        "share": mem.get("share"),
                        "share_reason": mem.get("share_reason"),
                        "final_choice": mem.get("final_choice"),
                        "final_reason": mem.get("final_reason"),
                        "decision_sequence": "; ".join(mem.get("decision_sequence", [])),
                        "total_revenue": float(mem.get("total_revenue", 0.0) or 0.0)
                    })

            # Firm temp_mem
            if "Firm" in line and "temp_mem:" in line:
                mem_match = re.search(r"Firm \d+ temp_mem: ({.*)", line)  # relaxed
                if mem_match:
                    mem = safe_eval_mem(mem_match.group(1))
                    firm_temp.append({
                        "firm_num": mem.get("num_firms", current_firm),
                        "round": current_round,
                        "firm_id": f"F{mem.get('index', -1)}",
                        "price": mem.get("price"),
                        "price_reason": mem.get("price_reason"),
                        "profit": mem.get("profit"),
                        "sale_num": mem.get("sale_num"),
                        "revenue": mem.get("revenue"),
                        "share_rate": mem.get("share_rate"),
                        "share_rate_predicted": mem.get("share_rate_predicted")
                    })

    df_consumer_temp = pd.DataFrame(consumer_temp)
    df_firm_temp = pd.DataFrame(firm_temp)

    # Handle detailed_data
    detailed_dir = os.path.join(os.path.dirname(log_file), "detailed_data")
    if os.path.exists(detailed_dir) and not df_consumer_temp.empty:
        detail_frames = []
        for fname in os.listdir(detailed_dir):
            if fname.endswith(".csv") and fname.startswith("detailed_exp"):
                df_detail = pd.read_csv(os.path.join(detailed_dir, fname))
                df_detail = df_detail[df_detail["Agent_Type"] == "Consumer"].copy()
                df_detail["consumer_id"] = "C" + (df_detail["Agent_Index"] + 1).astype(str)
                df_detail.rename(columns={"Round": "round"}, inplace=True)
                m = re.search(r"firms_(\d+)_", fname)
                if m:
                    df_detail["firm_num"] = int(m.group(1))
                detail_frames.append(
                    df_detail[["firm_num", "round", "consumer_id", "Privacy_Cost", "Valuations"]]
                )
        if detail_frames:
            df_consumer_temp = df_consumer_temp.merge(
                pd.concat(detail_frames, ignore_index=True),
                on=["firm_num", "round", "consumer_id"],
                how="left"
            )

    # Output directory
    os.makedirs(out_dir, exist_ok=True)
    consumer_csv = os.path.join(out_dir, "consumer_temp.csv")
    df_consumer_temp.to_csv(consumer_csv, index=False)
    firm_csv = os.path.join(out_dir, "firm_temp.csv")
    df_firm_temp.to_csv(firm_csv, index=False)

    # Split consumer by firm_num
    split_dir = os.path.join(out_dir, "consumer")
    os.makedirs(split_dir, exist_ok=True)
    for firm, group in df_consumer_temp.groupby("firm_num"):
        group.to_csv(os.path.join(split_dir, f"firm{firm}.csv"), index=False)

    return consumer_csv, firm_csv