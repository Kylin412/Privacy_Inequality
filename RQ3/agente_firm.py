# expert_labeling_csv.py
import ast
import json
import time
import traceback
from typing import Dict, Any, Tuple, List
import concurrent.futures

import pandas as pd
from openai import OpenAI

from configs.config_firm_agente import API_KEY, API_BASE_URL, MODEL_NAME, TEMPERATURE, EXPERT_PROMPT_TEMPLATE

# ===================== Manual Configurable Parameters =====================
INPUT_CSV = r"allgrok-firm/grok-firm_final.csv"
OUTPUT_CSV = "grok-firm_4.1+gemini_label.csv"
MAX_WORKERS = 5
PER_REQUEST_SLEEP = 0.0

# ===================== Basic Configuration =====================
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

FACTOR_NAMES = [
    "Profit Margin",
    "Customer Loyalty",
    "Competitive Pressure",
    "Customer Churn Risk",
    "Demand Uncertainty",
    "Net Benefit",
    "Profit Maximization",
    "Loss Minimization",
]

# ===================== Build Content for Prompt =====================
def build_input_and_output_from_row(row: Dict[str, Any]) -> Tuple[str, str]:
    # Construct raw input data (only retain price decision rationale)
    input_data = f"""
[Price Decision Rationale]
{row.get('price_reason')}
    """.strip()

    # Parse importance vector and adjacency matrix
    vec_raw = row.get("vector_intersection")
    mat_raw = row.get("matrix_intersection")

    try:
        vec = ast.literal_eval(vec_raw) if isinstance(vec_raw, str) else vec_raw
        mat = ast.literal_eval(mat_raw) if isinstance(mat_raw, str) else mat_raw
    except Exception as e:
        print(f"⚠️ Failed to parse vector/matrix (firm_id={row.get('firm_id')}): {e}")
        vec = [3] * len(FACTOR_NAMES)
        mat = [[0] * len(FACTOR_NAMES) for _ in range(len(FACTOR_NAMES))]

    # Build importance vector mapping
    importance_vector = {}
    for i, factor_name in enumerate(FACTOR_NAMES):
        try:
            importance_vector[factor_name] = int(vec[i])
        except Exception:
            importance_vector[factor_name] = 3

    # Build adjacency matrix mapping
    adjacency_matrix = {}
    for i, fi in enumerate(FACTOR_NAMES):
        row_map = {}
        try:
            row_i = mat[i]
        except Exception:
            row_i = [0] * len(FACTOR_NAMES)
        for j, fj in enumerate(FACTOR_NAMES):
            try:
                row_map[fj] = int(row_i[j])
            except Exception:
                row_map[fj] = 0
        adjacency_matrix[fi] = row_map

    output_data_obj = {
        "importance_vector": importance_vector,
        "adjacency_matrix": adjacency_matrix,
    }
    output_data = json.dumps(output_data_obj, ensure_ascii=False, indent=2)

    return input_data, output_data

# ===================== Call Expert Model for 0/1 Labeling =====================
def get_expert_label(input_data: str, output_data: str) -> Tuple[int, str]:
    prompt = EXPERT_PROMPT_TEMPLATE.format(
        input_data=input_data,
        output_data=output_data
    )

    max_retries = 3
    last_content = ""

    for attempt in range(max_retries):
        try:
            if PER_REQUEST_SLEEP > 0:
                time.sleep(PER_REQUEST_SLEEP)

            resp = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a rigorous corporate decision analysis expert."
                            "Only output 0 or 1, no other content."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content.strip()
            last_content = content

            if content in ("0", "1"):
                return int(content), content

            print(f"⚠️ Model response cannot be parsed to 0/1: {content!r}")

        except Exception as e:
            print(f"⚠️ API Error (Attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    return 0, last_content

# ===================== Concurrent Worker: Process Single Row =====================
def _worker(row_index_and_data: Tuple[int, Dict[str, Any]]) -> Tuple[int, int, str, str, str]:
    idx, row_dict = row_index_and_data
    firm_id = str(row_dict.get("firm_id", ""))

    try:
        input_data, output_data = build_input_and_output_from_row(row_dict)
        label, raw_reply = get_expert_label(input_data, output_data)
        return idx, label, firm_id, "", raw_reply
    except Exception:
        err_trace = traceback.format_exc()
        return idx, 0, firm_id, err_trace, ""

# ===================== Main Process =====================
def run_expert_labeling(input_csv: str, output_csv: str):
    print(f"📂 Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    total = len(df)
    print(f"🚀 Starting expert review for {total} records...")

    if total == 0:
        print("⚠️ CSV is empty, no data to process.")
        return

    labels: List[int] = [None] * total
    raw_replies: List[str] = [None] * total

    # Prepare task list
    tasks: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in df.iterrows():
        tasks.append((idx, row.to_dict()))

    done_count = 0

    # Process with thread pool
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_index = {
            executor.submit(_worker, task): task[0]
            for task in tasks
        }

        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            row_display = idx + 1
            try:
                row_idx, label, firm_id, err_trace, raw_reply = future.result()
                assert row_idx == idx

                labels[idx] = label
                raw_replies[idx] = raw_reply
                done_count += 1

                if err_trace:
                    print(f"⚠️ Error processing row {row_display} (firm_id={firm_id}), default label 0 assigned")
                    print("------ traceback ------")
                    print(err_trace)
                else:
                    preview = (raw_reply[:40] + "...") if isinstance(raw_reply, str) and len(raw_reply) > 40 else raw_reply
                    print(f"   [{done_count}/{total}] Row={row_display}, firm_id={firm_id}, Label={label}, RawReply={preview!r}")

            except Exception as e:
                print(f"⚠️ Exception in future.result() for row {row_display}: {e}")
                labels[idx] = 0
                raw_replies[idx] = ""
                done_count += 1

    # Add labeling results to DataFrame
    df["expert_label"] = labels
    df["expert_raw_reply"] = raw_replies

    print(f"💾 Writing results to: {output_csv}")
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("✅ Expert review completed.")

# ===================== Entry Point =====================
if __name__ == "__main__":
    run_expert_labeling(INPUT_CSV, OUTPUT_CSV)