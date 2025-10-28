import os
import json
import csv
import time
import sys
import traceback
from openai import OpenAI
from config_firm import API_KEY, API_BASE_URL, MODEL_NAME, TEMPERATURE, PROMPT_TEMPLATE

# Initialize client
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)


# === Model invocation ===
def call_model(prompt: str) -> str:
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a rigorous data analysis assistant. Output valid JSON format."},
                    {"role": "user", "content": prompt},
                ],
            )
            text = response.choices[0].message.content.strip()
            print(f"✅ Model invocation successful (attempt {attempt})")
            return text
        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                traceback.print_exc(file=sys.stdout)
                return f"[ERROR] {e}"
            time.sleep(2 * attempt)


# === Auxiliary functions ===

def extract_json(text: str):
    """Parse returned JSON"""
    try:
        data = json.loads(text)
        return {
            "result": data.get("result"),
            "Matrix": data.get("Matrix")
        }
    except json.JSONDecodeError:
        print("⚠️ Unable to parse JSON, outputting original text: ", text)
        return {"raw_reply": text}


def fill_prompt(headers, row, template):
    """Assemble prompt"""
    row_dict = {h.strip(): v for h, v in zip(headers, row)}
    formatted_row = ", ".join(f"{h}: {v}" for h, v in zip(headers, row))
    row_dict["formatted_row"] = formatted_row
    try:
        return template.format(** row_dict)
    except KeyError:
        return template.format(formatted_row=formatted_row)


def process_csv_once(input_path: str, output_path: str):
    """Single experiment: process CSV, invoke model, write JSONL"""
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    results = []
    for i, row in enumerate(rows, start=1):
        print(f"\n🌀 Experiment data row {i} ...")
        prompt = fill_prompt(headers, row, PROMPT_TEMPLATE)
        reply = call_model(prompt)
        result_extracted = extract_json(reply)
        results.append({"row": i, "result": result_extracted})

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Experiment output saved: {output_path}")


def load_jsonl(path):
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            row = obj["row"]
            matrix_raw = obj["result"].get("Matrix")
            matrix = json.loads(matrix_raw) if isinstance(matrix_raw, str) else matrix_raw
            data[row] = matrix
    return data


def intersection_matrix(matrices):
    """Support intersection operation for any number of input matrices"""
    n = len(matrices[0])
    num = len(matrices)
    return [
        [
            int(all(m[i][j] == 1 for m in matrices))
            for j in range(n)
        ]
        for i in range(n)
    ]


def intersection_for_experiments(exp_files, output_path):
    """Calculate intersection of multiple experiment results in a single loop and write to jsonl"""
    print(f"\n🎯 Calculating intersection: {exp_files}")
    all_data = [load_jsonl(f) for f in exp_files]
    row_keys = sorted(all_data[0].keys())

    results = []
    for row in row_keys:
        row_matrices = [data[row] for data in all_data]
        inter = intersection_matrix(row_matrices)
        results.append({"row": row, "Matrix": inter})

    with open(output_path, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Loop intersection results saved to: {output_path}")


def combine_all_loops(loop_files, final_output):
    """Aggregate intersection results from all loops into a single final file"""
    all_results = []
    for f in loop_files:
        with open(f, "r", encoding="utf-8") as r:
            for line in r:
                all_results.append(json.loads(line))

    with open(final_output, "w", encoding="utf-8") as w:
        for item in all_results:
            w.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"\n🏁 All loops completed! Final integrated file: {final_output}")


# === Main entry ===
def main():
    # ---- Parameter interface ----
    input_csv = "grok-gpt-frim.csv"       # Source CSV
    experiment_count = 5  # Number of experiments per loop
    loop_count = 1                 # Total number of loops

    # ---- Main process ----
    all_loop_results = []

    for loop_idx in range(1, loop_count + 1):
        print(f"\n🚀 Starting loop {loop_idx} =====================")

        exp_files = []
        for exp_idx in range(1, experiment_count + 1):
            output_path = f"output-loop{loop_idx}-exp{exp_idx}_firm.jsonl"
            process_csv_once(input_csv, output_path)
            exp_files.append(output_path)

        loop_output = f"intersection-loop{loop_idx}_firm.jsonl"
        intersection_for_experiments(exp_files, loop_output)
        all_loop_results.append(loop_output)

    final_output = "final_intersection_firm.jsonl"
    combine_all_loops(all_loop_results, final_output)


if __name__ == "__main__":
    main()