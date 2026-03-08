import os
import json
import argparse
from statistics import mean


def load_tasks(tasks_json_path):
    with open(tasks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    task_map = {}
    for item in data:
        task_id = str(item["task_id"])
        gold = set(item.get("related_attributes", []))
        task_map[task_id] = gold
    return task_map


def load_predicted_attributes(requirements_dir, task_id):
    path = os.path.join(requirements_dir, f"task_{task_id}_requirements.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if "required_attributes" in obj and isinstance(obj["required_attributes"], dict):
        return set(obj["required_attributes"].keys())

    if isinstance(obj, dict):
        return set(obj.keys())

    return set()


def prf(pred, gold):
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
        start = test_start or 0
        end = test_end if test_end is not None else len(task_ids)
        task_ids = task_ids[start:end]

    rows = []
    missing = []

    for task_id in task_ids:
        gold = gold_map[task_id]
        pred = load_predicted_attributes(requirements_dir, task_id)

        if pred is None:
            missing.append(task_id)
            continue

        result = prf(pred, gold)
        result["task_id"] = task_id
        result["gold"] = sorted(gold)
        result["pred"] = sorted(pred)
        rows.append(result)

    if not rows:
        raise ValueError("No prediction files found for evaluation.")

    summary = {
        "num_tasks_scored": len(rows),
        "num_missing_predictions": len(missing),
        "exact_match": mean(r["exact_match"] for r in rows),
        "precision_macro": mean(r["precision"] for r in rows),
        "recall_macro": mean(r["recall"] for r in rows),
        "f1_macro": mean(r["f1"] for r in rows),
        "jaccard_macro": mean(r["jaccard"] for r in rows),
        "avg_tp": mean(r["tp"] for r in rows),
        "avg_fp": mean(r["fp"] for r in rows),
        "avg_fn": mean(r["fn"] for r in rows),
        "missing_task_ids": missing,
    }

    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks_json_path", required=True)
    parser.add_argument("--requirements_dir", required=True)
    parser.add_argument("--test_start", type=int, default=None)
    parser.add_argument("--test_end", type=int, default=None)
    parser.add_argument("--save_details_path", default=None)
    args = parser.parse_args()

    summary, rows = evaluate(
        tasks_json_path=args.tasks_json_path,
        requirements_dir=args.requirements_dir,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    print(json.dumps(summary, indent=2))

    if args.save_details_path:
        with open(args.save_details_path, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "details": rows}, f, indent=2)


if __name__ == "__main__":
    main()
