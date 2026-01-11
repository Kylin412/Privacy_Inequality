import os
import json
from openai import OpenAI
from config_factor import SYSTEM_PROMPT, INDUCTIVE_PROMPT_TEMPLATE, INITIAL_FACTORS

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")


def run_inductive_pipeline(data_samples):
    """
    Implementation of the LLM-assisted inductive coding pipeline described in the paper.
    """
    print(f"--- Starting Inductive Coding Pipeline (N={len(data_samples)}) ---")

    initial_framework_str = json.dumps(INITIAL_FACTORS, indent=2)

    results_log = []

    for sample in data_samples:
        print(f"Processing Sample ID: {sample['id']}...")

        # 组装 Prompt
        user_content = INDUCTIVE_PROMPT_TEMPLATE.format(
            initial_framework=initial_framework_str,
            agent_rationale=sample['price_reason']
        )

        try:
            response = client.chat.completions.create(
                model="",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            # 解析结果
            raw_output = response.choices[0].message.content
            extracted_logic = json.loads(raw_output)

            # 存储结果
            results_log.append({
                "id": sample['id'],
                "extracted_logic": extracted_logic
            })

            print(f"Successfully extracted factors: {list(extracted_logic['discovered_factors'].values())}")

        except Exception as e:
            print(f"Error processing sample {sample['id']}: {e}")

    with open("inductive_coding_results.json", "w") as f:
        json.dump(results_log, f, indent=4)

    print("\n--- Pipeline Complete. Taxonomy results saved to inductive_coding_results.json ---")


if __name__ == "__main__":
    mock_agent_data = [
    ]

    run_inductive_pipeline(mock_agent_data)
