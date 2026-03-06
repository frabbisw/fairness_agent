# Setup & Run Guide — FairnessAgent (self-contained)

## What this folder contains

```
fairness_agent_clean/       ← copy this into your repo root
├── pipeline.py             # Main LangGraph agent pipeline
├── prompts.py              # All LLM prompts + ablation matrix
├── solar_bridge.py         # Reads Solar's output files (no subprocess in dev mode)
├── run_ablation.py         # Runs all ablation variants for paper comparison table
├── requirements.txt        # Python dependencies
└── SETUP.md                # This file
```

These files are **completely independent** of `generate_code/`, `agents/`,
or any other previously generated code. They only need:
- `dataset/prompts.jsonl`  (your dataset — already exists)
- `fairness_test/`         (Solar's test suite — already exists)
- `commands/agent_commands_bash.sh`  (Solar runner — already exists)

---

## Step 1 — Place the files

Copy this folder into your repo root:
```
fairness_agent/
├── fairness_agent_clean/   ← paste here
├── dataset/
├── fairness_test/
├── commands/
└── ...
```

---

## Step 2 — Create a fresh Conda environment

```bash
conda create -n fairness_agent python=3.11 -y
conda activate fairness_agent
```

> **Why Python 3.11?** LangGraph requires 3.10+. 3.11 is stable and
> compatible with all other dependencies in your repo.

---

## Step 3 — Install existing repo dependencies

```bash
cd fairness_agent/commands
bash setup.sh           # installs textX, pytest, and other Solar dependencies
cd ..
```

If `setup.sh` fails or doesn't exist, install Solar's dependencies manually:
```bash
pip install textx pytest openai anthropic python-dotenv
```

---

## Step 4 — Install the new agent dependencies

```bash
cd fairness_agent/fairness_agent_clean
pip install -r requirements.txt
```

This installs `langgraph`, `langchain`, `langchain-openai`, and `openai`.

---

## Step 5 — Set your OpenAI API key

Create a `.env` file in the repo root (NOT inside `fairness_agent_clean/`):

```bash
# fairness_agent/.env
OPENAI_API_KEY=sk-...your-key-here...
```

Add it to `.gitignore` so you never accidentally commit it:
```bash
echo ".env" >> .gitignore
```

---

## Step 6 — Run Solar first to generate bias_info files

**This step is required before running the agent pipeline.**
Solar must run on the developer's generated code first to produce
the `bias_info` and `related_info` files that the agent reads.

```bash
cd fairness_agent/commands

# Run Solar on tasks 0–9 (small test to verify it works)
bash gen_and_exp.sh \
  5 \
  1.0 \
  default \
  "../dataset/prompts.jsonl" \
  "../outputs/gpt_baseline" \
  gpt \
  0 \
  10
```

After this you should see:
```
outputs/gpt_baseline/
├── response/
│   └── developer/
│       ├── task_0_generated_code.jsonl
│       ├── task_1_generated_code.jsonl
│       └── ...
└── test_result/
    └── developer/
        └── bias_info_files/
            ├── bias_info0.jsonl       ← the agent reads this
            ├── related_info0.jsonl    ← and this
            └── ...
```

---

## Step 7 — Run the agent pipeline

```bash
cd fairness_agent/fairness_agent_clean

# CACHED MODE (uses Solar results from Step 6 — no subprocess, fast for dev)
python pipeline.py \
  --prompts_path   ../dataset/prompts.jsonl \
  --model_dir      ../outputs/gpt_baseline \
  --output_dir     ../outputs/agent_results \
  --test_start     0 \
  --test_end       10 \
  --max_iterations 3
```

You will see output like:
```
============================================================
Task 0  (0–10)
============================================================
  [Developer] task=0
  [Solar/developer] task=0: biased=[gender] | missing=[income]
  [Analyzer] task=0 iter=0
  [Repairer] task=0 iter=0
  [Solar/repairer_iter1] task=0: ✓ PASS
============================================================
Task 1  (0–10)
...
============================================================
DONE  passed=8/10  avg_iters=1.6
Results → ../outputs/agent_results
```

---

## Step 8 — Run with live Solar (for actual experiments)

In live mode, the pipeline calls Solar automatically after each repair
so it can loop until the code passes.

```bash
python pipeline.py \
  --prompts_path   ../dataset/prompts.jsonl \
  --model_dir      ../outputs/agent_live \
  --output_dir     ../outputs/agent_live/results \
  --solar_bash     ../commands/agent_commands_bash.sh \
  --data_path      ../dataset/prompts.jsonl \
  --num_samples    5 \
  --max_iterations 3 \
  --test_start     0 \
  --test_end       50
```

---

## Step 9 — Run the ablation study (for the paper)

```bash
python run_ablation.py \
  --prompts_path   ../dataset/prompts.jsonl \
  --model_dir      ../outputs/ablation_model \
  --output_dir     ../outputs/ablation_results \
  --solar_bash     ../commands/agent_commands_bash.sh \
  --num_samples    5 \
  --max_iterations 3 \
  --test_start     0 \
  --test_end       343
```

This runs 7 variants automatically and prints a comparison table:
```
=======================================================================
ABLATION RESULTS
=======================================================================
Variant                Label                                  Pass% AvgIter
-----------------------------------------------------------------------
baseline_default       No fairness hint (baseline)            45.2%    1.00
baseline_cot           CoT only (Solar paper baseline)        48.1%    1.00
baseline_pcot          P-CoT only (Solar paper baseline)      47.3%    1.00
agent_full             Full agent pipeline (ours)             81.5%    1.62
agent_no_analyzer      Ablation: no Analyzer                  63.4%    1.89
agent_min_repair       Ablation: minimal repair               71.2%    1.74
agent_struct_plan      Ablation: structured plan              78.9%    1.67
```

---

## Output files explained

After `pipeline.py` runs, each task produces a `task_N_result.json`:
```json
{
  "task_id": "0",
  "final_code": "def qualified_for_financial_aid(self) -> bool:\n    ...",
  "iterations_run": 2,
  "final_passed": true,
  "history": [
    {
      "iteration": 0,
      "code_before": "...",
      "solar_result": {
        "biased_attrs": ["gender"],
        "missing_attrs": ["income"],
        "passed": false
      },
      "analysis": "The code uses self.gender which causes unfair outcomes...",
      "code_after": "..."
    }
  ]
}
```

And a `summary.json`:
```json
{
  "total": 10,
  "passed": 8,
  "pass_rate": 0.8,
  "avg_iterations": 1.6,
  "test_range": [0, 10],
  "prompt_styles": {"developer": "agent", "analyzer": "agent", "repairer": "agent"}
}
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `ModuleNotFoundError: langgraph` | `pip install -r requirements.txt` |
| `ModuleNotFoundError: prompts` | Run from `fairness_agent_clean/` directory |
| `openai.AuthenticationError` | Check `.env` has `OPENAI_API_KEY=sk-...` |
| `FileNotFoundError: bias_info0.jsonl` | Run Solar (Step 6) first |
| `Solar bash failed` | Check path to `agent_commands_bash.sh` is correct |
| LangGraph version error | `pip install "langgraph>=0.2.0,<0.3.0"` |

---

## Quick reference

| Argument | Default | Description |
|---|---|---|
| `--prompts_path` | required | `dataset/prompts.jsonl` |
| `--model_dir` | required | Base dir for Solar I/O |
| `--output_dir` | required | Where agent results are written |
| `--solar_bash` | `""` | Path to bash runner. Empty = cached mode |
| `--num_samples` | `5` | Code samples per task |
| `--max_iterations` | `3` | Max repair loops |
| `--test_start` | `0` | First task index |
| `--test_end` | `10` | Last task index (exclusive) |
| `--developer_style` | `agent` | `agent / default / chain_of_thoughts / positive_chain_of_thoughts` |
| `--analyzer_style` | `agent` | `agent / structured` |
| `--repairer_style` | `agent` | `agent / minimal` |
