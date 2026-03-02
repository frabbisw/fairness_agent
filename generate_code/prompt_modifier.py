import json
import os
import sys
import time
from itertools import islice

from dotenv import load_dotenv
from openai import OpenAI

from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

# -------------------------
# Clients (kept consistent with your repo style)
# -------------------------
gpt_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


# -------------------------
# IO helpers
# -------------------------
def read_jsonl_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl_line(fp, obj: dict):
    json.dump(obj, fp, ensure_ascii=False)
    fp.write("\n")


# -------------------------
# Output cleaning
# -------------------------
def clean_model_prompt_output(text: str) -> str:
    """
    Model should output ONLY the rewritten prompt string (no JSON).
    This function strips common wrappers/fences safely.
    """
    if text is None:
        return ""
    t = text.strip()

    # remove code fences if the model wraps output
    if t.startswith("```"):
        t = t.replace("```python", "").replace("```", "").strip()

    # remove common labels
    for prefix in ["PROMPT:", "REWRITTEN PROMPT:", "REWRITTEN_PROMPT:", "OUTPUT:", "RESULT:"]:
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].strip()

    return t


# -------------------------
# LLM call (same switch pattern as developer.py)
# -------------------------
def prompt_conversation(system_style: str, user_q: str, temp: float, model_name: str) -> str:
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[
                {"role": "system", "content": system_style},
                {"role": "user", "content": user_q},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    elif model_name == "llama":
        response = llama_client.chat.completions.create(
            model="CODELLAMA/CODELLAMA-70B-INSTRUCT-HF",
            temperature=temp,
            messages=[
                {"role": "system", "content": system_style},
                {"role": "user", "content": user_q},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    elif model_name == "bison":
        parameters = {
            "temperature": temp,
            "max_output_tokens": 1024,
        }
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=system_style)
        response = chat.send_message(user_q, **parameters)
        return (response.text or "").strip()

    elif model_name == "claude":
        # keep pause to reduce rate-limit failures (consistent with your agents)
        time.sleep(20)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=temp,
            system=system_style,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_q}]}],
        )
        return (response.content[0].text or "").strip()

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")


# -------------------------
# Prompt styles
# -------------------------
prompt_styles = {
    "gpt": {
        "agent": (
            '''You are a prompt-modifier for fairness testing.
Rewrite the prompt by removing all attributes irrelevant to the task (especially demographic/protected ones).
Preserve task semantics and method signatures.
Do not add new attributes or requirements.
Output only the rewritten prompt text. No JSON. No explanations.'''
        ),
    },
    "llama": {
        "default": (
            "You are a prompt-modifier agent for fairness testing. "
            "Rewrite the prompt to REMOVE irrelevant attributes and their mentions. "
            "Preserve the task meaning and method signature.\n"
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
        "agent": (
            "Rewrite prompt to remove irrelevant attributes for fairness. "
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
    },
    "bison": {
        "default": (
            "You are a prompt-modifier agent for fairness testing. "
            "Rewrite the prompt to REMOVE irrelevant attributes and their mentions. "
            "Preserve the task meaning and method signature.\n"
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
        "agent": (
            "Rewrite prompt to remove irrelevant attributes for fairness. "
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
    },
    "claude": {
        "default": (
            "You are a prompt-modifier agent for fairness testing. "
            "Rewrite the prompt to REMOVE irrelevant attributes and their mentions. "
            "Preserve the task meaning and method signature.\n"
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
        "agent": (
            "Rewrite prompt to remove irrelevant attributes for fairness. "
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
        ),
    },
}


def build_system_style(base_style: str) -> str:
    if context_message:
        return base_style + "\n\nCONTEXT:\n" + context_message.strip()
    return base_style


def build_user_query(original_prompt: str) -> str:
    return (
        "ORIGINAL_PROMPT:\n"
        f"{original_prompt}\n\n"
        "Rewrite it following the rules. Output ONLY the rewritten prompt text."
    )


def modify_prompts(
    input_prompts_path: str,
    output_prompts_path: str,
    iterations: int,
    temperature: float,
    style: str,
    model_name: str,
    test_start: int,
    test_end: int,
):
    """
    Reads JSONL prompts with {"task_id":..., "prompt":...}
    Writes JSONL prompts with the SAME format, but prompt rewritten by the LLM.
    iterations is accepted to mirror other agents; we generate 1 rewrite per task.
    """
    os.makedirs(os.path.dirname(output_prompts_path), exist_ok=True)

    with open(output_prompts_path, "w", encoding="utf-8") as out_f:
        for index, json_obj in enumerate(
            islice(read_jsonl_file(input_prompts_path), test_start, test_end),
            start=test_start,
        ):
            print(f"Processing line {index}")
            print("-" * 50)

            task_id = json_obj.get("task_id", "default")
            original_prompt = json_obj.get("prompt", "")

            if not isinstance(original_prompt, str) or not original_prompt.strip():
                write_jsonl_line(
                    out_f,
                    {"task_id": task_id, "prompt": original_prompt if isinstance(original_prompt, str) else ""},
                )
                continue

            system_style = style
            user_q = build_user_query(original_prompt)

            raw = prompt_conversation(system_style, user_q, temperature, model_name)
            rewritten_prompt = clean_model_prompt_output(raw)

            # Fallback if model returns empty/garbage
            if not rewritten_prompt.strip():
                rewritten_prompt = original_prompt

            write_jsonl_line(out_f, {"task_id": task_id, "prompt": rewritten_prompt})

            print(f"[OK] task_id={task_id} rewritten_len={len(rewritten_prompt)}")
            print("-" * 100)

    print(f"[DONE] Wrote modified prompts to: {output_prompts_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prompt modifier agent (flag-based).")

    p.add_argument(
        "--jsonl_input_file_path",
        required=True,
        type=str,
        help="Path to input JSONL prompts file (each line: {task_id, prompt}).",
    )
    p.add_argument(
        "--output_prompt_filename",
        required=True,
        type=str,
        help=(
            "Output filename (e.g., prompts_modified.jsonl). "
            "If a bare filename is provided, it is saved alongside the input file."
        ),
    )
    p.add_argument("--num_samples", required=True, type=int, help="Kept for interface compatibility; rewrite once per task.")
    p.add_argument("--temperature", required=True, type=float, help="Sampling temperature.")
    p.add_argument("--prompt_style", required=True, type=str, help="Prompt style key (e.g., default/agent).")
    p.add_argument("--model_name", required=True, type=str, choices=["gpt", "llama", "bison", "claude"], help="Which model client to use.")
    p.add_argument("--test_start", required=True, type=int, help="Start index (inclusive) in the JSONL.")
    p.add_argument("--test_end", required=True, type=int, help="End index (exclusive) in the JSONL.")
    return p


if __name__ == "__main__":
    print("starting prompt_modifier agent")
    print("=" * 50)

    parser = build_arg_parser()
    args = parser.parse_args()

    input_jsonl = args.jsonl_input_file_path
    output_path = resolve_output_path(input_jsonl, args.output_prompt_filename)

    print("input_jsonl:", input_jsonl)
    print("output_path:", output_path)
    print("num_samples:", args.num_samples)
    print("temperature:", args.temperature)
    print("prompt_style:", args.prompt_style)
    print("model_name:", args.model_name)
    print("test_start:", args.test_start)
    print("test_end:", args.test_end)

    base_style = prompt_styles.get(args.model_name, {}).get(args.prompt_style)
    if base_style is None:
        raise ValueError(
            f"Unknown prompt_style '{args.prompt_style}' for model_name '{args.model_name}'. "
            f"Available: {list(prompt_styles.get(args.model_name, {}).keys())}"
        )

    modify_prompts(
        input_prompts_path=input_jsonl,
        output_prompts_path=output_path,
        iterations=args.num_samples,
        temperature=args.temperature,
        style=base_style,
        model_name=args.model_name,
        test_start=args.test_start,
        test_end=args.test_end,
    )
