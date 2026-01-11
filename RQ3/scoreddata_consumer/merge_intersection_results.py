import csv
import json

def merge_by_row(main_csv, intersection_csv, output_csv):
    # === 1. 读取交集结果 ===
    intersection_data = {}
    with open(intersection_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row_id = int(row["row"])
            except ValueError:
                continue
            intersection_data[row_id] = {
                "vector": row.get("vector", ""),
                "matrix": row.get("matrix", "")
            }

    # === 2. 读取主文件 ===
    with open(main_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    # === 3. 准备新表头 ===
    new_headers = headers + ["vector_intersection", "matrix_intersection"]

    # === 4. 拼接数据 ===
    merged_rows = []
    for idx, row in enumerate(rows, start=1):
        # 这里假设 idx（行号）即对应 row 值
        # 如果你的主表中某列是明确 row 字段，也可以用 row_dict["row"] 解析匹配
        row_number = idx  # 第2行代表 row=1，以此类推
        if row_number in intersection_data:
            vec = intersection_data[row_number]["vector"]
            mat = intersection_data[row_number]["matrix"]
        else:
            vec = ""
            mat = ""
        merged_rows.append(row + [vec, mat])

    # === 5. 写出新文件 ===
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(merged_rows)

    print(f"✅ 合并完成，输出文件：{output_csv}")


# === 主入口 ===
if __name__ == "__main__":
    # 示例：
    # main_csv: 原始文件（带表头）
    # intersection_csv: 上一步生成的 loop1_intersection.csv
    merge_by_row("allgrok.csv", "loop1_majority_vote_1.csv", "allgrok_final_1.csv")