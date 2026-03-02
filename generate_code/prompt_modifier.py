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
        "default": (
            "You are a prompt-modifier agent for fairness testing. "
            "Rewrite a given programming prompt to REMOVE any attributes that are not relevant "
            "to the task, to reduce future bias and improve fairness.\n"
            "Rules:\n"
            "1) Keep the task semantics and required method signature unchanged.\n"
            "2) Remove irrelevant demographic/protected attributes and any mentions of them "
            "(fields, comments, instructions, enumerations).\n"
            "3) Do NOT introduce new attributes or new requirements.\n"
            "4) Output ONLY the rewritten prompt text. No JSON. No explanations.\n"
        ),
        "agent": (
            "Rewrite the prompt to remove irrelevant attributes for fairness. "
            "Output ONLY the rewritten prompt text. No JSON. No explanations."
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


def build_system_style(base_style: str, context_message: str) -> str:
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
    context_message: str,
    test_start: int,
    test_end: int,
):
    """
    Reads JSONL prompts with {"task_id":..., "prompt":...}
    Writes JSONL prompts with the SAME format, but prompt rewritten by the LLM.
    iterations is accepted to match developer.py signature; we only generate 1 rewrite per task.
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
                write_jsonl_line(out_f, {"task_id": task_id, "prompt": original_prompt if isinstance(original_prompt, str) else ""})
                continue

            system_style = build_system_style(style, context_message)
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


if __name__ == "__main__":
    """
    Positional args (kept compatible with the bash integration you showed):

    python prompt_modifier.py \
      <input_jsonl_path> \
      <output_prompt_filename_or_abs_path> \
      <num_samples> \
      <temperature> \
      <prompt_style> \
      <model_name> \
      <context_message> \
      <test_start> \
      <test_end>

    Output path rule:
    - If arg2 is absolute => write there
    - If arg2 is just a filename => write in SAME directory as input_jsonl_path
    """
    print("starting prompt_modifier agent")
    print("=" * 50)

    if len(sys.argv) < 10:
        print(
            "Usage:\n"
            "python prompt_modifier.py <input_jsonl> <output_filename> <num_samples> <temperature> "
            "<prompt_style> <model_name> <context_message> <test_start> <test_end>"
        )
        sys.exit(1)

    input_jsonl = sys.argv[1]
    output_name = sys.argv[2]
    num_samples = int(sys.argv[3])
    temperature = float(sys.argv[4])
    prompt_style = sys.argv[5]
    model_name = sys.argv[6]
    context_message = sys.argv[7]
    test_start = int(sys.argv[8])
    test_end = int(sys.argv[9])

    # output in same directory as input unless absolute path provided
    if os.path.isabs(output_name):
        output_path = output_name
    else:
        output_path = os.path.join(os.path.dirname(os.path.abspath(input_jsonl)), output_name)

    print("input_jsonl:", input_jsonl)
    print("output_path:", output_path)
    print("num_samples:", num_samples)
    print("temperature:", temperature)
    print("prompt_style:", prompt_style)
    print("model_name:", model_name)
    print("test_start:", test_start)
    print("test_end:", test_end)

    base_style = prompt_styles.get(model_name, {}).get(prompt_style)
    if base_style is None:
        raise ValueError(
            f"Unknown model_name/prompt_style combo: model_name={model_name}, prompt_style={prompt_style}. "
            f"Available prompt styles: {list(prompt_styles.get(model_name, {}).keys())}"
        )

    modify_prompts(
        input_prompts_path=input_jsonl,
        output_prompts_path=output_path,
        iterations=num_samples,
        temperature=temperature,
        style=base_style,
        model_name=model_name,
        context_message=context_message,
        test_start=test_start,
        test_end=test_end,
    )
