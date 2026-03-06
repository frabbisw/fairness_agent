"""
pipeline.py
===========
LangGraph multi-agent pipeline for social bias mitigation.

Full pipeline (what this file handles end-to-end):
─────────────────────────────────────────────────────────────────
  Step 1  Developer LLM  → generates code from prompt
  Step 2  Solar bash     → tests developer code, writes bias_info files
  Step 3  Analyzer LLM   → reads bias_info, writes natural-language repair plan
  Step 4  Repairer LLM   → rewrites code following the plan
  Step 5  Solar bash     → tests repaired code, writes bias_info files
          └─ if bias remains and iterations left → back to Step 3
          └─ if passed or max iterations → END
─────────────────────────────────────────────────────────────────

Solar bash call signature (from agent_commands_bash.sh):
  bash agent_commands_bash.sh \
    $1=SAMPLING  $2=TEMPERATURE  $3=PROMPT_STYLE  $4=DATA_PATH \
    $5=MODEL_DIR  $6=MODEL_NAME  $7=TEST_START  $8=TEST_END

Solar directory structure written by bash script:
  MODEL_DIR/response/developer/task_<id>_generated_code.jsonl
  MODEL_DIR/test_result/developer/bias_info_files/bias_info<id>.jsonl
  MODEL_DIR/test_result/developer/bias_info_files/related_info<id>.jsonl
  MODEL_DIR/response/repairer/task_<id>_generated_code.jsonl
  MODEL_DIR/test_result/repairer/bias_info_files/bias_info<id>.jsonl
  MODEL_DIR/test_result/repairer/bias_info_files/related_info<id>.jsonl

Run (full fresh run, 5 tasks):
  cd fairness_agent_clean
  python pipeline.py \
    --prompts_path   ../dataset/prompts_sample.jsonl \
    --model_dir      ../outputs/agent_test \
    --output_dir     ../outputs/agent_test/results \
    --solar_bash     ../commands/agent_commands_bash.sh \
    --model_name     gpt \
    --sampling       5 \
    --test_start     0 \
    --test_end       5 \
    --max_iterations 3
"""

import json
import os
from pathlib import Path
from typing import TypedDict

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

from prompts import get_prompt
from solar_bridge import (
    SolarResult,
    write_code_for_solar,
    read_solar_result,
    run_solar_bash,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Shared state — every node reads and writes this
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # ── Fixed for the whole run ──────────────────────────────────────────
    task_id:        str
    prompt:         str
    model_dir:      str
    output_dir:     str
    solar_bash:     str      # path to agent_commands_bash.sh
    data_path:      str      # path to prompts JSONL (passed to Solar bash)
    model_name:     str      # "gpt" (used by Solar bash for dir naming)
    sampling:       int
    temperature:    float
    max_iterations: int
    prompt_styles:  dict     # {"developer": str, "analyzer": str, "repairer": str}

    # ── Updated each iteration ───────────────────────────────────────────
    iteration:      int
    current_code:   str
    solar_result:   dict     # SolarResult.to_dict()
    analysis:       str      # Analyzer's repair plan

    # ── Append-only log ──────────────────────────────────────────────────
    history:        list


# ---------------------------------------------------------------------------
# LLM — only GPT active; extend here for other models
# ---------------------------------------------------------------------------
def make_llm(temperature: float) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Node 1: Developer
# Calls the LLM to generate code from the prompt.
# Writes code to MODEL_DIR/response/developer/ so Solar can test it.
# ---------------------------------------------------------------------------
def developer_node(state: AgentState) -> AgentState:
    print(f"  [Developer] task={state['task_id']}")

    system = get_prompt("developer", state["prompt_styles"]["developer"])
    resp = make_llm(state["temperature"]).invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["prompt"]),
    ])
    code = resp.content.strip()

    # Write to Solar's expected location
    write_code_for_solar(
        code=code,
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase="developer",
        num_samples=state["sampling"],
    )

    return {**state, "current_code": code, "iteration": 0}


# ---------------------------------------------------------------------------
# Node 2: Solar (deterministic tool — no LLM)
# Calls the Solar bash pipeline then reads its output files.
# Phase is "developer" on iteration 0, "repairer" on all subsequent iterations.
# ---------------------------------------------------------------------------
def solar_node(state: AgentState) -> AgentState:
    phase = "developer" if state["iteration"] == 0 else "repairer"
    task_id = state["task_id"]

    print(f"  [Solar/{phase}] task={task_id} — running bash pipeline...")

    run_solar_bash(
        solar_bash=state["solar_bash"],
        data_path=state["data_path"],
        model_dir=state["model_dir"],
        model_name=state["model_name"],
        sampling=state["sampling"],
        temperature=state["temperature"],
        prompt_style=state["prompt_styles"]["developer"],  # Solar uses this for dir label
        test_start=int(task_id),
        test_end=int(task_id) + 1,
    )

    result: SolarResult = read_solar_result(
        task_id=task_id,
        model_dir=state["model_dir"],
        phase=phase,
        sample_index=0,
    )

    print(f"  [Solar/{phase}] task={task_id}: {result.summary()}")
    return {**state, "solar_result": result.to_dict()}


# ---------------------------------------------------------------------------
# Node 3: Analyzer (genuine review — fault localization, no code writing)
# Reads Solar test results and writes a natural-language repair plan.
# ---------------------------------------------------------------------------
def analyzer_node(state: AgentState) -> AgentState:
    solar = state["solar_result"]

    if solar["passed"]:
        return {**state, "analysis": "PASS"}

    print(f"  [Analyzer] task={state['task_id']} iter={state['iteration']}")

    biased_str  = ", ".join(solar["biased_attrs"])  or "none"
    missing_str = ", ".join(solar["missing_attrs"]) or "none"

    system = get_prompt("analyzer", state["prompt_styles"]["analyzer"])
    user_msg = (
        f"CODE:\n{state['current_code']}\n\n"
        f"BIASED ATTRIBUTES (caused Solar test failures):\n{biased_str}\n\n"
        f"MISSING TASK-RELEVANT ATTRIBUTES (absent from code):\n{missing_str}\n\n"
        "Write a repair plan."
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    return {**state, "analysis": resp.content.strip()}


# ---------------------------------------------------------------------------
# Node 4: Repairer
# Rewrites the method guided by the Analyzer's plan.
# Writes repaired code to MODEL_DIR/response/repairer/ so Solar can re-test.
# ---------------------------------------------------------------------------
def repairer_node(state: AgentState) -> AgentState:
    if state["analysis"] == "PASS":
        return state

    print(f"  [Repairer] task={state['task_id']} iter={state['iteration']}")

    system = get_prompt("repairer", state["prompt_styles"]["repairer"])
    user_msg = (
        f"CURRENT CODE:\n{state['current_code']}\n\n"
        f"REPAIR PLAN:\n{state['analysis']}"
    )
    resp = make_llm(temperature=0.2).invoke([
        SystemMessage(content=system),
        HumanMessage(content=user_msg),
    ])
    repaired = resp.content.strip()

    # Write to Solar's expected location for repairer phase
    write_code_for_solar(
        code=repaired,
        task_id=state["task_id"],
        model_dir=state["model_dir"],
        phase="repairer",
        num_samples=state["sampling"],
    )

    # Log this iteration
    entry = {
        "iteration":    state["iteration"],
        "code_before":  state["current_code"],
        "solar_result": state["solar_result"],
        "analysis":     state["analysis"],
        "code_after":   repaired,
    }

    return {
        **state,
        "current_code": repaired,
        "history": list(state.get("history", [])) + [entry],
    }


# ---------------------------------------------------------------------------
# Node 5: Increment — advance iteration counter after each repair cycle
# ---------------------------------------------------------------------------
def increment_node(state: AgentState) -> AgentState:
    return {**state, "iteration": state["iteration"] + 1}


# ---------------------------------------------------------------------------
# Routing — called after every solar_node
# ---------------------------------------------------------------------------
def route_after_solar(state: AgentState) -> str:
    if state["solar_result"]["passed"]:
        return "end"
    if state["iteration"] >= state["max_iterations"]:
        return "end"   # max iterations reached, accept current code
    return "analyze"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(AgentState)

    g.add_node("developer", developer_node)
    g.add_node("solar",     solar_node)
    g.add_node("analyzer",  analyzer_node)
    g.add_node("repairer",  repairer_node)
    g.add_node("increment", increment_node)

    g.set_entry_point("developer")
    g.add_edge("developer",  "solar")

    g.add_conditional_edges(
        "solar",
        route_after_solar,
        {"analyze": "analyzer", "end": END},
    )

    g.add_edge("analyzer",  "repairer")
    g.add_edge("repairer",  "increment")
    g.add_edge("increment", "solar")   # re-test; routing handles exit

    return g.compile()


# ---------------------------------------------------------------------------
# Run pipeline on a list of tasks
# ---------------------------------------------------------------------------
def run_pipeline(
    prompts_path: str,
    model_dir: str,
    output_dir: str,
    solar_bash: str,
    model_name: str = "gpt",
    sampling: int = 5,
    temperature: float = 1.0,
    prompt_styles: dict = None,
    max_iterations: int = 3,
    test_start: int = 0,
    test_end: int = 5,
) -> list[dict]:

    if prompt_styles is None:
        prompt_styles = {
            "developer": "agent",
            "analyzer":  "agent",
            "repairer":  "agent",
        }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Load only the tasks we need
    with open(prompts_path, encoding="utf-8") as f:
        all_tasks = [json.loads(line) for line in f if line.strip()]
    tasks = all_tasks[test_start:test_end]

    if not tasks:
        print(f"[ERROR] No tasks found in range [{test_start}, {test_end})")
        return []

    graph = build_graph()
    results = []

    for task in tasks:
        task_id = str(task.get("task_id", "?"))
        prompt  = task.get("prompt", "")

        print(f"\n{'='*60}")
        print(f"Task {task_id}  (index {test_start}–{test_end})")
        print(f"{'='*60}")

        initial: AgentState = {
            "task_id":        task_id,
            "prompt":         prompt,
            "model_dir":      model_dir,
            "output_dir":     output_dir,
            "solar_bash":     solar_bash,
            "data_path":      prompts_path,
            "model_name":     model_name,
            "sampling":       sampling,
            "temperature":    temperature,
            "max_iterations": max_iterations,
            "prompt_styles":  prompt_styles,
            "iteration":      0,
            "current_code":   "",
            "solar_result":   {
                "passed": False,
                "biased_attrs": [],
                "missing_attrs": [],
            },
            "analysis": "",
            "history":  [],
        }

        final = graph.invoke(initial)

        result = {
            "task_id":        task_id,
            "final_code":     final["current_code"],
            "iterations_run": final["iteration"] + 1,
            "final_passed":   final["solar_result"]["passed"],
            "history":        final["history"],
        }
        results.append(result)

        # Save per-task result
        out_file = Path(output_dir) / f"task_{task_id}_result.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  → saved {out_file.name}")

    # Aggregate summary
    total  = len(results)
    passed = sum(1 for r in results if r["final_passed"])
    summary = {
        "total":          total,
        "passed":         passed,
        "pass_rate":      round(passed / total, 3) if total else 0,
        "avg_iterations": round(
            sum(r["iterations_run"] for r in results) / total, 2
        ) if total else 0,
        "test_range":    [test_start, test_end],
        "prompt_styles": prompt_styles,
    }
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE  passed={passed}/{total}  avg_iters={summary['avg_iterations']}")
    print(f"Results → {output_dir}")
    print(f"{'='*60}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="FairnessAgent — full pipeline: Developer→Solar→Analyzer→Repairer→Solar"
    )
    p.add_argument("--prompts_path",    required=True,
                   help="Path to prompts JSONL file (e.g. dataset/prompts_sample.jsonl)")
    p.add_argument("--model_dir",       required=True,
                   help="Base output dir for Solar I/O (e.g. outputs/agent_test)")
    p.add_argument("--output_dir",      required=True,
                   help="Where to write per-task result JSON files")
    p.add_argument("--solar_bash",      required=True,
                   help="Path to commands/agent_commands_bash.sh")
    p.add_argument("--model_name",      default="gpt",
                   help="Model label passed to Solar bash (default: gpt)")
    p.add_argument("--sampling",        type=int,   default=5,
                   help="Number of code samples per task (default: 5)")
    p.add_argument("--temperature",     type=float, default=1.0,
                   help="LLM temperature for developer (default: 1.0)")
    p.add_argument("--max_iterations",  type=int,   default=3,
                   help="Max Analyzer→Repairer repair loops (default: 3)")
    p.add_argument("--test_start",      type=int,   default=0,
                   help="First task index in prompts file (default: 0)")
    p.add_argument("--test_end",        type=int,   default=5,
                   help="Last task index exclusive (default: 5)")
    p.add_argument("--developer_style", default="agent",
                   choices=["agent","default","chain_of_thoughts",
                            "positive_chain_of_thoughts"])
    p.add_argument("--analyzer_style",  default="agent",
                   choices=["agent","structured"])
    p.add_argument("--repairer_style",  default="agent",
                   choices=["agent","minimal"])
    args = p.parse_args()

    run_pipeline(
        prompts_path=args.prompts_path,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        solar_bash=args.solar_bash,
        model_name=args.model_name,
        sampling=args.sampling,
        temperature=args.temperature,
        prompt_styles={
            "developer": args.developer_style,
            "analyzer":  args.analyzer_style,
            "repairer":  args.repairer_style,
        },
        max_iterations=args.max_iterations,
        test_start=args.test_start,
        test_end=args.test_end,
    )
