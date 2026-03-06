"""
solar_bridge.py
===============
Reads Solar's existing output files produced by the Solar bash pipeline.

What Solar already produces (from parse_bias_info.py in your repo):
  <model_dir>/test_result/<phase>/bias_info_files/bias_info<task_id>.jsonl
  <model_dir>/test_result/<phase>/bias_info_files/related_info<task_id>.jsonl

Each file has one JSON object per sample (iteration), e.g.:
  {"bias_info": "gender,age"}        ← attributes that failed tests
  {"related_info": "income"}         ← task-relevant attrs missing from code
  {"bias_info": "none"}              ← no bias found

This module ONLY reads those files — it does NOT re-run Solar or call any LLM.
Solar is a deterministic tool; we treat its outputs as ground truth.

Two-phase usage in the pipeline
--------------------------------
Phase "developer" : Solar results after the Developer generates code
Phase "repairer"  : Solar results after the Repairer fixes code
                    We use "repairer_iter{N}" to keep iterations separate.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SolarResult:
    task_id:       str
    phase:         str
    sample_index:  int                 # which iteration (0-based)
    biased_attrs:  list[str] = field(default_factory=list)
    missing_attrs: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.biased_attrs) == 0 and len(self.missing_attrs) == 0

    def summary(self) -> str:
        if self.passed:
            return "✓ PASS"
        parts = []
        if self.biased_attrs:
            parts.append(f"biased=[{', '.join(self.biased_attrs)}]")
        if self.missing_attrs:
            parts.append(f"missing=[{', '.join(self.missing_attrs)}]")
        return " | ".join(parts)

    def to_dict(self) -> dict:
        return {
            "task_id":      self.task_id,
            "phase":        self.phase,
            "sample_index": self.sample_index,
            "biased_attrs": self.biased_attrs,
            "missing_attrs":self.missing_attrs,
            "passed":       self.passed,
        }


def _parse_attr_string(value: str) -> list[str]:
    """Parse Solar's comma-separated attribute string. 'none' → []."""
    if not value or value.strip().lower() == "none":
        return []
    return [a.strip() for a in value.split(",") if a.strip()]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def read_solar_result(
    task_id: str,
    model_dir: str,
    phase: str,
    sample_index: int = 0,
) -> SolarResult:
    """
    Read Solar's bias_info and related_info for a specific task + phase + sample.

    Args:
        task_id:      The task ID string (e.g. "0", "42")
        model_dir:    Base output directory (same as Solar's model_dir)
        phase:        "developer" | "repairer" | "repairer_iter1" etc.
        sample_index: Which sample line to read (0-based)

    Returns:
        SolarResult with biased_attrs and missing_attrs populated.
    """
    base = Path(model_dir) / "test_result" / phase / "bias_info_files"
    bias_path    = base / f"bias_info{task_id}.jsonl"
    related_path = base / f"related_info{task_id}.jsonl"

    bias_records    = _read_jsonl(bias_path)
    related_records = _read_jsonl(related_path)

    biased  = []
    missing = []

    if sample_index < len(bias_records):
        biased = _parse_attr_string(bias_records[sample_index].get("bias_info", "none"))

    if sample_index < len(related_records):
        missing = _parse_attr_string(related_records[sample_index].get("related_info", "none"))

    return SolarResult(
        task_id=task_id,
        phase=phase,
        sample_index=sample_index,
        biased_attrs=biased,
        missing_attrs=missing,
    )


def write_code_for_solar(
    code: str,
    task_id: str,
    model_dir: str,
    phase: str,
    sample_index: int = 0,
    num_samples: int = 1,
) -> Path:
    """
    Write generated code to the path Solar expects, so Solar can test it.

    Solar expects:
      <model_dir>/response/<phase>/task_<task_id>_generated_code.jsonl
    with one {"generated_code": "..."} per line (one per sample).
    """
    path = Path(model_dir) / "response" / phase / f"task_{task_id}_generated_code.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write num_samples lines (Solar tests each sample; we repeat same code)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            json.dump({"generated_code": code}, f, ensure_ascii=False)
            f.write("\n")

    return path
