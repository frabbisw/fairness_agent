"""
run_ablation.py
===============
Runs every variant in prompts.ABLATION_MATRIX and prints a comparison table.
Use this to generate the results for your paper's ablation section.

Usage
-----
  python run_ablation.py \
    --prompts_path   ../dataset/prompts.jsonl \
    --model_dir      ../outputs/ablation/model_dir \
    --output_dir     ../outputs/ablation \
    --test_start     0 \
    --test_end       50 \
    --max_iterations 3

    # add --solar_bash ../commands/agent_commands_bash.sh for live Solar
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
    solar_bash: str = "",
    data_path: str = "",
    num_samples: int = 5,
    max_iterations: int = 3,
    test_start: int = 0,
    test_end: int = 10,
) -> dict:

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summaries = {}

    for key, (dev_style, ana_style, rep_style, label) in ABLATION_MATRIX.items():
        print(f"\n{'#'*65}")
        print(f"# VARIANT: {label}")
        print(f"#   developer={dev_style} | analyzer={ana_style} | repairer={rep_style}")
        print(f"{'#'*65}")

        variant_output = str(Path(output_dir) / key)
        variant_model  = str(Path(model_dir)  / key)

        # Baselines have no Analyzer/Repairer; one iteration only
        is_baseline    = ana_style is None
        effective_iters = 1 if is_baseline else max_iterations

        results = run_pipeline(
            prompts_path=prompts_path,
            model_dir=variant_model,
            output_dir=variant_output,
            solar_bash=solar_bash,
            data_path=data_path or prompts_path,
            prompt_styles={
                "developer": dev_style,
                "analyzer":  ana_style or "agent",   # fallback; won't be called for baselines
                "repairer":  rep_style or "agent",
            },
            num_samples=num_samples,
            max_iterations=effective_iters,
            test_start=test_start,
            test_end=test_end,
        )

        total  = len(results)
        passed = sum(1 for r in results if r["final_passed"])
        avg_it = round(sum(r["iterations_run"] for r in results) / total, 2) if total else 0

        summaries[key] = {
            "label":      label,
            "passed":     passed,
            "total":      total,
            "pass_rate":  round(passed / total, 3) if total else 0,
            "avg_iters":  avg_it,
            "styles": {
                "developer": dev_style,
                "analyzer":  ana_style,
                "repairer":  rep_style,
            },
        }

    # ── Print comparison table ──────────────────────────────────────────────
    print(f"\n\n{'='*75}")
    print("ABLATION RESULTS")
    print(f"{'='*75}")
    print(f"{'Variant':<22} {'Label':<38} {'Pass%':>6} {'AvgIter':>8}")
    print("-" * 75)
    for k, s in summaries.items():
        print(f"{k:<22} {s['label']:<38} "
              f"{s['pass_rate']*100:>5.1f}% {s['avg_iters']:>8.2f}")

    # ── Save full summary ───────────────────────────────────────────────────
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
    p.add_argument("--solar_bash",     default="",
                   help="Path to agent_commands_bash.sh. Empty = cached mode.")
    p.add_argument("--data_path",      default="")
    p.add_argument("--num_samples",    type=int, default=5)
    p.add_argument("--max_iterations", type=int, default=3)
    p.add_argument("--test_start",     type=int, default=0)
    p.add_argument("--test_end",       type=int, default=10)
    args = p.parse_args()

    run_all_ablations(
        prompts_path=args.prompts_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        solar_bash=args.solar_bash,
        data_path=args.data_path,
        num_samples=args.num_samples,
        max_iterations=args.max_iterations,
        test_start=args.test_start,
        test_end=args.test_end,
    )
