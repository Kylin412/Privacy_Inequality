# config.py
# 定义初始理论因子 (Grounded in behavioral decision theory)
INITIAL_FACTORS = {
    "Benefits": ["Search Cost Savings"],
    "Drawbacks": ["Privacy Loss"],
    "Trade-offs": ["Profit Maximization","Loss Minimization"]
}
SYSTEM_PROMPT = "You are a senior researcher specializing in Behavioral Decision Theory and Data Economics."
# 任务提示词：要求 LLM 根据原始文本验证并补全因子
INDUCTIVE_PROMPT_TEMPLATE = """
Role: You are an expert researcher in Behavioral Economics and Data Privacy.
Task: Perform an Inductive Coding process to extract decision factors from LLM Agent rationales.

Context: We are building a taxonomy of factors influencing data market decisions. 
Our initial framework (theoretically grounded) includes:
{initial_framework}

Input Rationale:
"{agent_rationale}"

Instructions:
1. Analyze the input rationale to identify decision drivers.
2. Validate if the initial factors exist in this rationale.
3. Inductively identify any emerging factors not covered by the initial list.
4. Categorize all identified factors into three dimensions: Benefits, Drawbacks, and Trade-offs.

Output Format (JSON):
{{
    "validated_factors": [],
    "emerging_factors": [],
    "final_taxonomy_segment": {{
        "Benefits": [],
        "Drawbacks": [],
        "Trade-offs": []
    }},
    "evidence_quote": "..."
}}
"""