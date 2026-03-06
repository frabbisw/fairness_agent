"""
solar_bridge.py
===============
Reads and writes files in the exact structure Solar's bash pipeline uses.

Confirmed from agent_commands_bash.sh:

  MODEL_DIR/
  ├── response/
  │   ├── developer/
  │   │   └── task_<id>_generated_code.jsonl   ← developer writes here
  │   └── repairer/
  │       └── task_<id>_generated_code.jsonl   ← repairer writes here
  └── test_result/
      ├── developer/
      │   └── bias_info_files/
      │       ├── bias_info<id>.jsonl           ← Solar writes, we read
      │       └── related_info<id>.jsonl        ← Solar writes, we read
      └── repairer/
          └── bias_info_files/
              ├── bias_info<id>.jsonl
              └── related_info<id>.jsonl

bias_info<id>.jsonl — one JSON line per sample:
    {"bias_info": "gender,age"}   → attributes that failed Solar tests
    {"bias_info": "none"}         → no bias

related_info<id>.jsonl — one JSON line per sample:
    {"related_info": "income"}    → task-relevant attrs missing from code
    {"related_info": "none"}      → all present
"""

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data class representing one Solar test result
# ---------------------------------------------------------------------------
@dataclass
class SolarResult:
    task_id:      str
    phase:        str           # "developer" or "repairer"
    biased_attrs: list[str] = field(default_factory=list)
    missing_attrs: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.biased_attrs and not self.missing_attrs

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
            "biased_attrs": self.biased_attrs,
            "missing_attrs":self.missing_attrs,
            "passed":       self.passed,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_attrs(value: str) -> list[str]:
    """'gender,age' → ['gender','age'].  'none'/'' → []."""
    if not value or value.strip().lower() == "none":
        return []
    return [a.strip() for a in value.split(",") if a.strip()]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


# ---------------------------------------------------------------------------
# Write code so Solar can test it
# ---------------------------------------------------------------------------
def write_code_for_solar(
    code: str,
    task_id: str,
    model_dir: str,
    phase: str,
    num_samples: int = 5,
) -> Path:
    """
    Write generated/repaired code to the path Solar expects.
    Solar reads: MODEL_DIR/response/<phase>/task_<id>_generated_code.jsonl
    One JSON line per sample (Solar tests each sample independently).
    We write the same code for all samples.
    """
    path = (Path(model_dir) / "response" / phase
            / f"task_{task_id}_generated_code.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _ in range(num_samples):
            json.dump({"generated_code": code}, f, ensure_ascii=False)
            f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Read Solar's output after it has run
# ---------------------------------------------------------------------------
def read_solar_result(
    task_id: str,
    model_dir: str,
    phase: str,          # "developer" or "repairer"
    sample_index: int = 0,
) -> SolarResult:
    """
    Read Solar's bias_info and related_info for one task.
    sample_index selects which sample line to read (0 = first sample).
    """
    base = Path(model_dir) / "test_result" / phase / "bias_info_files"
    bias_rows    = _read_jsonl(base / f"bias_info{task_id}.jsonl")
    related_rows = _read_jsonl(base / f"related_info{task_id}.jsonl")

    biased = (
        _parse_attrs(bias_rows[sample_index].get("bias_info", "none"))
        if sample_index < len(bias_rows) else []
    )
    missing = (
        _parse_attrs(related_rows[sample_index].get("related_info", "none"))
        if sample_index < len(related_rows) else []
    )

    return SolarResult(
        task_id=task_id,
        phase=phase,
        biased_attrs=biased,
        missing_attrs=missing,
    )


# ---------------------------------------------------------------------------
# Run Solar's bash pipeline for one phase
# ---------------------------------------------------------------------------
def run_solar_bash(
    solar_bash: str,
    data_path: str,
    model_dir: str,
    model_name: str,
    sampling: int,
    temperature: float,
    prompt_style: str,
    test_start: int,
    test_end: int,
) -> bool:
    """
    Call agent_commands_bash.sh with the exact argument order it expects:
      $1=SAMPLING $2=TEMPERATURE $3=PROMPT_STYLE $4=DATA_PATH
      $5=MODEL_DIR $6=MODEL_NAME $7=TEST_START $8=TEST_END

    Returns True if the command succeeded.
    """
    cmd = [
        "bash", solar_bash,
        str(sampling),
        str(temperature),
        prompt_style,
        data_path,
        model_dir,
        model_name,
        str(test_start),
        str(test_end),
    ]
    print(f"  [Solar bash] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [WARN] Solar bash returned code {result.returncode}")
        return False
    return True
