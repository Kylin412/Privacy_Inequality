# config.py
import os

# 密钥
API_KEY = os.getenv("GROK_API_KEY")

# API 服务器地址
API_BASE_URL = ""

# 选择模型名：推理 or 非推理
MODEL_NAME = "grok-4-fast"

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
| Major Category                                                                | Factor                   | Definition                                                                                                            | Influence on Sharing Decision                                                                                                                                                            | Influence on Purchase Decision                                                                                                                                  |
| ----------------------------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Benefits** (Positive drivers: promote sharing and encourage purchasing)     | Search Cost Savings      | Reduction in search cost due to personalized recommendations after sharing                                            | High savings promote sharing (“significant reduction...outweighs privacy” → `share=True`); low savings suppress sharing (“minimal...privacy outweighs” → `share=False`).                 | High savings promote `BestProfit` and encourage purchase (“positive 0.21...purchase”); low savings push search/exit (“minimal...negative...leave”).             |
|                                                                               | Positive Profit          | Perceived positive profit                                                                                             | Positive profit promotes sharing (“expect positive...outweighs privacy” → `share=True`); lack of positive profit suppresses sharing (“no positive...privacy outweighs” → `share=False`). | Positive profit promotes purchase (“positive 0.23...buy”); lack of profit drives search/exit (“no positive...leave”).                                           |
| **Drawbacks** (Negative inhibitors: suppress sharing and promote search/exit) | Privacy Loss             | Value and immediate perception of privacy loss                                                                        | High loss suppresses sharing (“privacy 0.051 incurs...outweighs” → `share=False`); low loss promotes sharing (“low privacy 0.026...share” → `share=True`).                               | High loss + negative profit drive exit (“privacy 0.055...negative...leave”); low loss promotes purchase (“low privacy...positive...buy”).                       |
|                                                                               | Negative Profit          | Perceived negative profit                                                                                             | Negative profit suppresses sharing (“expect negative...privacy outweighs” → `share=False`); no negative profit promotes sharing (“no negative...share” → `share=True`).                  | Negative profit drives exit (“negative...leave”); no negative profit promotes purchase (“positive...buy”); uncertainty drives search (“zero...search further”). |
|                                                                               | Search Continuation Cost | Additional cost of continued search (e.g., “incurs 0.02 per search...no improvement”)                                 | High cost promotes sharing (“share to avoid 0.02 costs” → `share=True`); low cost reduces sharing incentive.                                                                             | High cost promotes purchase/exit (“incurs 0.02...purchase positive”); low cost promotes search (“low cost...search further”).                                   |
| **Trade-offs** (Integrative: determine joint behavior)                        | Net Benefit              | Compare savings vs. privacy cost to calculate `BestProfit` net value (e.g., “outweighs” + “positive/negative profit”) | Positive net promotes sharing (“significant outweighs privacy...positive net” → `share=True`); negative net suppresses sharing (“privacy outweighs...negative net” → `share=False`).     | Positive net benefit promotes purchase (“positive 0.23...purchase”); negative net benefit drives exit/search (“negative -0.14...leave”).                        |
|                                                                               | Profit Maximization      | Integration of pros and cons based on “maximize revenue”                                                              | Rational behavior promotes sharing (“rational to share...maximize” → `share=True`); suppresses sharing (“not share to maximize”).                                                        | Rational behavior promotes purchase (“rational...buy positive”); drives exit (“rational...leave”).                                                              |
|                                                                               | Loss Minimization        | Integration of pros and cons based on “minimize loss”                                                                 | High loss suppresses sharing (“rational to minimize...not share” → `share=False`); low loss promotes sharing (“no loss...share”).                                                        | Rational behavior drives exit/search (“minimize loss...leave”; “zero...search further”).                                                                        |
{
  "result": "<your extracted importance vector>",
  "Matrix": "<your extracted full adjacency matrix>"
}
"""
