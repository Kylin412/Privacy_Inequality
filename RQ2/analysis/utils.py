import ast
import numpy as np
from scipy import stats


class NumpyNumberCleaner(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            if node.func.value.id == "np" and node.args:
                if node.func.attr in ("float64", "int64"):
                    return self.visit(node.args[0])
        return self.generic_visit(node)


def safe_eval_mem(mem_str: str):
    """Enhanced: safely parse log content (robust + truncate + strip np.float64/int64 wrappers)"""
    if not mem_str:
        return {}

    # 1. Truncate to the last '}' to avoid concatenated Firm output
    if "}" in mem_str:
        mem_str = mem_str[: mem_str.rfind("}") + 1]

    try:
        # 2. AST parse + clean numpy type wrappers
        tree = ast.parse(mem_str, mode="eval")
        tree = NumpyNumberCleaner().visit(tree)
        ast.fix_missing_locations(tree)

        # 3. literal_eval to dict
        return ast.literal_eval(tree.body)

    except Exception as e:
        # Fault tolerance: avoid crashing the parser
        print(f"[WARN] safe_eval_mem parse failed: {e}\nraw: {mem_str[:200]}...")
        return {}


def mean_ci(arr, confidence=0.95):
    """Compute mean and CI half-width"""
    arr = np.array(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, 0.0
    mean = np.mean(arr)
    if n > 1:
        std = np.std(arr, ddof=1)
        sem = std / np.sqrt(n)
        z = stats.norm.ppf((1 + confidence) / 2.0)
        h = z * sem
    else:
        h = 0.0
    return mean, h