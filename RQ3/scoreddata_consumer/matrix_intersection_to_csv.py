import json
import csv
import glob
from collections import Counter


# === 1. 读取每个 jsonl 文件中的矩阵和向量 ===
def load_jsonl(path):
    """
    输入一个实验文件路径，
    返回 { row: {"vector": [...], "matrix": [[...], ...]} } 的字典。
    若某行存在错误或无法解析，自动跳过。
    """
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                row = obj.get("row")
                res = obj.get("result", {})

                # 跳过报错或异常的行
                if "raw_reply" in res:
                    continue

                vec_raw = res.get("result")
                mat_raw = res.get("Matrix")

                # 向量和矩阵可能是字符串，需要再次解析
                vector = json.loads(vec_raw) if isinstance(vec_raw, str) else vec_raw
                matrix = json.loads(mat_raw) if isinstance(mat_raw, str) else mat_raw

                # 基础校验
                if not isinstance(vector, list) or not isinstance(matrix, list):
                    continue

                data[row] = {"vector": vector, "matrix": matrix}
            except Exception as e:
                print(f"⚠️ 解析 {path} 某行失败：{e}")
    return data


# === 2. 向量取众数 ===
def consensus_vector(vectors):
    """
    对同行的多个实验向量逐元素取众数；
    若有平票情况，取最小值。
    """
    if not vectors:
        return []
    length = len(vectors[0])
    result = []
    for i in range(length):
        vals = [v[i] for v in vectors if i < len(v)]
        if not vals:
            result.append(None)
            continue
        counter = Counter(vals)
        max_count = max(counter.values())
        # 如果多个众数并列，选最小的（保证确定性）
        candidates = [k for k, c in counter.items() if c == max_count]
        result.append(min(candidates))
    return result


# === 3. 矩阵多数投票（多个实验） ===
def majority_vote_matrix(matrices):
    """
    支持任意数量的同维度方阵；
    同一位置有2个或以上的1就最终取1，否则0。
    """
    if not matrices:
        return []
    n = len(matrices[0])
    # 先校验所有矩阵维度一致
    for m in matrices:
        if len(m) != n or any(len(row) != n for row in m):
            raise ValueError("所有矩阵必须是同维度的方阵")

    # 逐位置统计1的数量，≥2则取1，否则0
    return [
        [int(sum(m[i][j] == 1 for m in matrices) >= 2) for j in range(n)]
        for i in range(n)
    ]


# === 4. 汇总所有实验，按 row 求投票结果和众数 ===
def compute_majority_vote_and_consensus(file_pattern, output_csv):
    """
    file_pattern: 如 "output-loop1-exp*.jsonl"
    output_csv: 输出 csv 文件路径
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        print("❌ 未找到匹配文件。")
        return
    print(f"🧩 检测到实验文件: {files}")

    all_data = [load_jsonl(f) for f in files]

    # 全部 row 的集合，并集求齐
    all_rows = sorted(set().union(*[set(d.keys()) for d in all_data]))

    results = []
    for row in all_rows:
        row_vectors = []
        row_matrices = []
        for d in all_data:
            if row in d:
                row_vectors.append(d[row]["vector"])
                row_matrices.append(d[row]["matrix"])

        if not row_vectors:
            continue  # 所有文件均缺此行，则跳过

        vec_result = consensus_vector(row_vectors)
        # 替换为多数投票逻辑
        mat_result = majority_vote_matrix(row_matrices)
        results.append({"row": row, "vector": vec_result, "matrix": mat_result})

    # 写出 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "vector", "matrix"])
        for item in results:
            writer.writerow([
                item["row"],
                json.dumps(item["vector"], ensure_ascii=False),
                json.dumps(item["matrix"], ensure_ascii=False)
            ])

    print(f"✅ 已输出结果至 {output_csv}")


# === 5. 主入口 ===
if __name__ == "__main__":
    # 例子模式：当前目录下的所有 output-loop1-exp*.jsonl
    compute_majority_vote_and_consensus("output-loop1-exp*.jsonl", "loop1_majority_vote_1.csv")