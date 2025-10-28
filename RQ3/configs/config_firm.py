# config.py
import os

# 密钥
API_KEY = os.getenv("GROK_API_KEY")

# API 服务器地址
API_BASE_URL = ""

# 选择模型名：推理 or 非推理
MODEL_NAME = "grok-4-fast"       # 或者 "grok-4-fast-reasoning"

# 模型参数
TEMPERATURE = 0.05

# 提示模板
PROMPT_TEMPLATE = """
You are a structured data analysis assistant.
Please read the following single line of CSV data and summarize two outputs according to our factor rules below:

1. Steps

Step 1: For each Factor, list the evidence found in the file.
Step 2: Based on the evidence, evaluate and assign an importance ranking (ties allowed, three levels in total:
1 = directly drives decision,
2 = secondary influence,
3 = marginal impact).

The generated ranking must include 1, 2, and 3, with roughly equal proportions.

Step 3: Output a pure vector (no extra text), with a fixed dimension equal to the number of factors (8D), formatted as:
[1,2,1,3,2,1,3,2]

The order of factors follows the table below.

2. Network Analysis

Based on the above factor analysis, first list the relationships supported by evidence, then generate an 8×8 undirected adjacency matrix (the matrix must be symmetric), representing the logical relationships between factors.

Values:
0 = no significant logical relationship
1 = clear logical relationship

Format:
[[0,1,0,...],[1,0,...],...]

Rows correspond to factor pairs, with both rows and columns ordered according to the table below.
| Major Category                                                                               | Factor               | Definition                                                         | Influence on Pricing Decision (High/Mid/Low Pricing)                                                                                                             |
| -------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Benefits** (Positive drivers: push high pricing and attract sharing consumers)             | Profit Margin        | Increase in unit profit from higher pricing                        | High margin drives high price (“high margin...maximizes profit” → high); low margin drives low price (“low margin...reduce price to attract” → low).             |
|                                                                                              | Customer Loyalty     | Dependence of data-sharing consumers on recommendations            | Loyalty drives high price (“loyal sharing base...sustain high price” → high); low loyalty drives low price (“no loyalty...lower price to compete” → low).        |
| **Drawbacks** (Negative inhibitors: suppress high pricing and push lower or adjusted prices) | Competitive Pressure | Market competition leading to price wars                           | High competition suppresses high prices (“intense competition...reduce price” → low); low competition allows high prices (“no competition...high price” → high). |
|                                                                                              | Customer Churn Risk  | Risk of customers leaving due to high pricing                      | High risk suppresses high price (“leave risk...lower price” → low); low risk allows high price (“low leave risk...high price” → high).                           |
|                                                                                              | Demand Uncertainty   | Demand fluctuation leading to pricing errors                       | High uncertainty suppresses high price (“uncertain...adjust to mid price” → mid); low uncertainty drives high price (“stable demand...high price” → high).       |
| **Trade-offs** (Integrative: determine pricing and joint outcomes)                           | Net Profit           | Compare pricing returns vs. cost risk to calculate expected profit | Positive net drives high price (“high outweighs low risk...positive net” → high); negative net drives low price (“competition outweighs...negative net” → low).  |
|                                                                                              | Profit Maximization  | Integration of pros and cons based on “rational...maximize profit” | Drives high price (“rational to high price...maximize” → high); suppresses high price (“not high to maximize” → low).                                            |
|                                                                                              | Loss Minimization    | Integration of pros and cons based on “rational...minimize loss”   | Suppresses high price (“rational to minimize...not high price” → low); drives high price (“no loss...high price”).                                               |
Please output only JSON in the following format:
{
  "result": "<your extracted importance vector>",
  "Matrix": "<your extracted full adjacency matrix>"
}

"""
