# Privacy_Inequality

## Main Simulation Experiment

This directory contains the **core simulation environment** used to evaluate LLM-driven consumer and firm behavior under different market competition settings. This component is associated with **RQ1/RQ2** in the paper.

---

### Overview

The simulation models a data market in which:

- **Consumer agents** decide whether to **share personal data**.
- **Firm agents** choose **pricing strategies** based on consumer behavior.
- Market outcomes (sharing ratio, equilibrium prices, consumer/firm surplus, welfare metrics) are evaluated across multiple **LLM-based decision policies**.

This folder contains the complete implementation of:

| File | Purpose |
|------|---------|
| `agents_complete.py` | Defines the **agent decision logic**, including consumer and firm behavior models driven by LLM or baseline strategies. |
| `rec_complete_9.10.py` | Main experiment runner. Executes full simulations across market sizes, model choices, and competition conditions. |
| `configs/` | Contains model configuration files, including LLM model names, API call parameters, and prompting templates. |

---

### Environment Requirements

This experiment requires access to one or more LLM APIs (e.g., OpenAI, Gemini, DeepSeek, xAI).  
To install Python dependencies:

```bash
pip install agentscope
```
By modifying the model configuration and parameters to run experiments, you can obtain simulation data of the multi-agent framework for this economic scenario.


## RQ1: Behavioral and Welfare Deviations from BNE under Competition

This directory contains the analysis workflow and artifacts for **Research Question 1 (RQ1)**.  
The goal is to quantify how LLM-driven agents deviate from the **Bayesian Nash Equilibrium (BNE)** across market competition levels, and whether such deviations amplify inequality or enhance consumer welfare.

This component focuses on **equilibrium alignment**, **market-competition sensitivity**, and **welfare decomposition**.

---

### Environment Setup

This module uses Python with `pip` dependency management.

To install required packages:

```bash
pip install pandas numpy matplotlib jupyter
```

File Structure:

```bash
RQ1/
│
├── final_results/              # Aggregated outputs (tables/figures) used in the paper draft
├── ICDE_Analysis.ipynb         # Main analysis notebook for RQ1 plots/tables
└── .keep
```

Typical Usage:

```bash
# Launch the notebook to reproduce MAE vs. BNE, CI bands, and welfare panels
jupyter notebook RQ1/ICDE_Analysis.ipynb
```

## RQ2: Heterogeneous Agents and Consumer Welfare under Competition

This directory contains the analysis pipeline and plotting scripts for **Research Question 2 (RQ2)**.  
We examine heterogeneous deployments where firms/consumers may use **rational agents**, **strong LLMs**, or **weak LLMs**, and assess how these capability mixes impact **aggregate consumer surplus** across competition levels.

This component focuses on **heterogeneity**, **competition–welfare interaction**, and **reproducible plotting**.

---

### Environment Setup

This module uses Python with `pip` dependency management.

To install required packages:

```bash
pip install pandas numpy matplotlib
```

File Structure:

```bash
RQ2/
│
├── analysis/                   # Intermediate data and helper scripts (inputs to plotting)
├── BNE/                        # BNE benchmark references for comparison
├── plot_analysis_results/      # Output figures for heterogeneous settings
├── all_results_old/            # Archived/legacy results (kept for provenance)
├── main_plot_analysis.py       # Entry script to generate RQ2 plots/tables
└── .keep
```

Typical Usage:

```bash
# Generate all figures/tables into plot_analysis_results/
python RQ2/main_plot_analysis.py
```



## RQ3: Factor Importance Scoring and Draft Bar Charts

This directory contains the analysis pipeline(CV-DLS) and intermediate results used in **Research Question 3 (RQ3)**.  
The goal of RQ3 is to identify and compare the **decision-driving factors** for consumer agents and firm agents under different model configurations.

This component focuses on **explainability**, **interpretation**, and **factor importance scoring**.

---

### Environment Setup

This module uses Python with `pip` dependency management.

To install required packages:

```bash
pip install pandas numpy matplotlib agentscope
```
File Structure:
```bash
RQ3/
│
├── Bar chart plotting/          # Draft bar charts (not final paper plots)
├── configs/                     # Keyword → factor configuration files
│
├── human_annotation/        # Instructions for human validation and result data
│
├── scoredata_consumer/          # Scoring results for consumer agents
├── scoredata_strong_firm/       # Scoring results for strong-model firm agents
├── scoredata_weak_firm/         # Scoring results for weak-model firm agents
│
├── agente_consumer.py           # Expert Validation for consumer data
├── agente_firm.py               # Expert Validation for firm data
├── score_consumer.py            # Computes factor scores for consumer agents
├── score_firm.py                # Computes factor scores for firm agents
└── .keep
```

Typical Usage:

```bash
python RQ3/score_consumer.py
python RQ3/score_firm.py
```

