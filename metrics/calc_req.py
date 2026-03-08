import argparse
import json
import os
from statistics import mean


def load_tasks(tasks_json_path):
    with open(tasks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("tasks.json must be a list of task objects.")

    task_map = {}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue

        task_id = str(idx)
        gold = set(item.get("related_attributes", []))
        task_map[task_id] = gold

    return task_map


def load_prediction(requirements_dir, task_id):
    path = os.path.join(requirements_dir, f"task_{task_id}_requirements.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "required_attributes" in data:
        required = data["required_attributes"]
        if isinstance(required, dict):
            return set(required.keys())
        if isinstance(required, list):
            return set(required)

    if isinstance(data, dict):
        return set(data.keys())

    return set()


def compute_metrics(pred, gold):
    tp = len(pred & gold)
    fp = len(pred - gold)
    fn = len(gold - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    exact = 1.0 if pred == gold else 0.0
    jaccard = tp / len(pred | gold) if (pred | gold) else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact,
        "jaccard": jaccard,
    }


def evaluate(tasks_json_path, requirements_dir, test_start=None, test_end=None):
    gold_map = load_tasks(tasks_json_path)

    task_ids = sorted(gold_map.keys(), key=lambda x: int(x))

    if test_start is not None or test_end is not None:
        start = test_start if test_start is not None else 0
        end = test_end if test_end is not None else len(task_ids)
        task_ids = task_ids[start:end]

    rows = []
    missing = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for task_id in task_ids:
        gold = gold_map[task_id]
        pred = load_prediction(requirements_dir, task_id)

        if pred is None:
            missing.append(task_id)
            continue

        result = compute_metrics(pred, gold)
        result["task_id"] = task_id
        result["gold"] = sorted(gold)
        result["pred"] = sorted(pred)
        rows.append(result)

        total_tp += result["tp"]
        total_fp += result["fp"]
        total_fn += result["fn"]

    if not rows:
        raise ValueError("No prediction files found for evaluation.")

    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    summary = {
        "num_tasks_total": len(task_ids),
        "num_tasks_scored": len(rows),
        "num_missing_predictions": len(missing),
        "exact_match_rate": mean(r["exact_match"] for r in rows),
        "precision_macro": mean(r["precision"] for r in rows),
        "recall_macro": mean(r["recall"] for r in rows),
        "f1_macro": mean(r["f1"] for r in rows),
        "jaccard_macro": mean(r["jaccard"] for r in rows),
        "precision_micro": micro_precision,
        "recall_micro": micro_recall,
        "f1_micro": micro_f1,
        "avg_tp": mean(r["tp"] for r in rows),
        "avg_fp": mean(r["fp"] for r in rows),
        "avg_fn": mean(r["fn"] for r in rows),
        "missing_task_ids": missing,
    }

    return summary, rows


def print_summary(summary):
    print("\nRequirement Analyst Evaluation\n")
    print(f"num_tasks_total:         {summary['num_tasks_total']}")
    print(f"num_tasks_scored:        {summary['num_tasks_scored']}")
    print(f"num_missing_predictions: {summary['num_missing_predictions']}")
    print(f"exact_match_rate:        {summary['exact_match_rate']:.4f}")
    print(f"precision_macro:         {summary['precision_macro']:.4f}")
    print(f"recall_macro:            {summary['recall_macro']:.4f}")
    print(f"f1_macro:                {summary['f1_macro']:.4f}")
    print(f"jaccard_macro:           {summary['jaccard_macro']:.4f}")
    print(f"precision_micro:         {summary['precision_micro']:.4f}")
    print(f"recall_micro:            {summary['recall_micro']:.4f}")
    print(f"f1_micro:                {summary['f1_micro']:.4f}")
    print(f"avg_tp:                  {summary['avg_tp']:.4f}")
    print(f"avg_fp:                  {summary['avg_fp']:.4f}")
    print(f"avg_fn:                  {summary['avg_fn']:.4f}")

    if summary["missing_task_ids"]:
        print("\nMissing task ids:")
        print(", ".join(summary["missing_task_ids"]))


def save_details(save_path, summary, rows):
    out = {
        "summary": summary,
        "details": rows,
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_json_path", required=True)
    parser.add_argument("--requirements_dir", required=True)
    parser.add_argument("--test_start", type=int, default=None)
    parser.add_argument("--test_end", type=int, default=None)
    parser.add_argument("--save_details_path", default=None)
    parser.add_argument("--show_errors", action="store_true")
    args = parser.parse_args()

    summary, rows = evaluate(
        tasks_json_path=args.tasks_json_path,
        requirements_dir=args.requirements_dir,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    print_summary(summary)

    if args.show_errors:
        print("\nPer-task mismatches:\n")
        for row in rows:
            if set(row["gold"]) != set(row["pred"]):
                print(f"task_id: {row['task_id']}")
                print(f"gold: {row['gold']}")
                print(f"pred: {row['pred']}")
                print()

    if args.save_details_path:
        save_details(args.save_details_path, summary, rows)
        print(f"Saved details to: {args.save_details_path}")


if __name__ == "__main__":
    main()
