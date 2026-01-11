import csv
import json

def safe_float(value):
    """尝试安全地转换为浮点数，失败或为空返回 None"""
    if value in (None, "", "null", "None"):
        return None
    try:
        return float(value)
    except:
        return None

def parse_vector(vec_str):
    """解析 vector_intersection 字符串为数字列表"""
    if not vec_str or str(vec_str).strip() in ("", "null", "None"):
        return []
    try:
        if isinstance(vec_str, str):
            return json.loads(vec_str)
        return vec_str
    except Exception:
        return []

def add_labels(input_csv, output_csv):
    with open(input_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["label1", "label2", "label3"]

        rows_out = []

        for row in reader:
            # --- label1: 隐私成本(Privacy_Cost) ≤ 0.04 ---
            privacy_cost = safe_float(row.get("Privacy_Cost"))
            if privacy_cost is None:
                label1 = ""
            else:
                label1 = privacy_cost <= 0.04

            # --- label2: firm_num ≤ 5 ---
            firm_num = safe_float(row.get("firm_num"))
            if firm_num is None:
                label2 = ""
            else:
                label2 = firm_num <= 5

            # --- label3: 比较 vector_intersection 第7与第8维 ---
            vec = parse_vector(row.get("vector_intersection"))
            if len(vec) >= 8:
                v7, v8 = vec[6], vec[7]
                if v7 < v8:
                    label3 = "less"
                elif v7 == v8:
                    label3 = "equal"
                else:
                    label3 = "greater"
            else:
                label3 = ""

            # 添加到行中
            row["label1"] = label1
            row["label2"] = label2
            row["label3"] = label3
            rows_out.append(row)

    # --- 写出结果 ---
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"✅ 处理完成：新文件 {output_csv}")

# === 主执行入口 ===
if __name__ == "__main__":
    add_labels("allgrok_final.csv", "allgrok_final_with_labels.csv")
