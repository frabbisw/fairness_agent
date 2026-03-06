"""
run_ablation.py
===============
Runs every variant in ABLATION_MATRIX and prints a comparison table.

Usage:
  python run_ablation.py \
    --prompts_path  ../dataset/prompts_sample.jsonl \
    --model_dir     ../outputs/ablation \
    --output_dir    ../outputs/ablation/results \
    --solar_bash    ../commands/agent_commands_bash.sh \
    --test_start    0 \
    --test_end      5
"""

import argparse
import json
from pathlib import Path

from pipeline import run_pipeline
from prompts import ABLATION_MATRIX


def run_all_ablations(
    prompts_path: str,
    model_dir: str,
    output_dir: str,
    solar_bash: str,
    model_name: str = "gpt",
    sampling: int = 5,
    temperature: float = 1.0,
    max_iterations: int = 3,
    test_start: int = 0,
    test_end: int = 5,
) -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summaries = {}

    for key, (dev_style, ana_style, rep_style, label) in ABLATION_MATRIX.items():
        print(f"\n{'#'*65}")
        print(f"# VARIANT : {label}")
        print(f"# Styles  : developer={dev_style} | analyzer={ana_style} | repairer={rep_style}")
        print(f"{'#'*65}")

        # Baselines have no Analyzer/Repairer — only 1 iteration
        is_baseline    = ana_style is None
        effective_iters = 1 if is_baseline else max_iterations

        results = run_pipeline(
            prompts_path=prompts_path,
            model_dir=str(Path(model_dir) / key),
            output_dir=str(Path(output_dir) / key),
            solar_bash=solar_bash,
            model_name=model_name,
            sampling=sampling,
            temperature=temperature,
            prompt_styles={
                "developer": dev_style,
                "analyzer":  ana_style or "agent",
                "repairer":  rep_style or "agent",
            },
            max_iterations=effective_iters,
            test_start=test_start,
            test_end=test_end,
        )

        total  = len(results)
        passed = sum(1 for r in results if r["final_passed"])
        avg_it = round(
            sum(r["iterations_run"] for r in results) / total, 2
        ) if total else 0

        summaries[key] = {
            "label":     label,
            "passed":    passed,
            "total":     total,
            "pass_rate": round(passed / total, 3) if total else 0,
            "avg_iters": avg_it,
        }

    # Print table
    print(f"\n\n{'='*72}")
    print("ABLATION RESULTS")
    print(f"{'='*72}")
    print(f"{'Variant':<22} {'Label':<38} {'Pass%':>6} {'AvgIter':>7}")
    print("-" * 72)
    for k, s in summaries.items():
        print(f"{k:<22} {s['label']:<38} "
              f"{s['pass_rate']*100:>5.1f}% {s['avg_iters']:>7.2f}")

    # Save
    out = Path(output_dir) / "ablation_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nFull summary → {out}")

    return summaries


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run all ablation variants")
    p.add_argument("--prompts_path",   required=True)
    p.add_argument("--model_dir",      required=True)
    p.add_argument("--output_dir",     required=True)
    p.add_argument("--solar_bash",     required=True)
    p.add_argument("--model_name",     default="gpt")
    p.add_argument("--sampling",       type=int,   default=5)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--max_iterations", type=int,   default=3)
    p.add_argument("--test_start",     type=int,   default=0)
    p.add_argument("--test_end",       type=int,   default=5)
    args = p.parse_args()

    run_all_ablations(
        prompts_path=args.prompts_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        solar_bash=args.solar_bash,
        model_name=args.model_name,
        sampling=args.sampling,
        temperature=args.temperature,
        max_iterations=args.max_iterations,
        test_start=args.test_start,
        test_end=args.test_end,
    )
