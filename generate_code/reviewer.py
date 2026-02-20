# generate_code/reviewer.py
import json
import os
import re
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import datastore
from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

# --- Clients: keep same style as developer.py ---
gpt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url="https://api.together.xyz/v1",
)

# Vertex / Google
google_client = datastore.Client()

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


# --- Helpers ---
def read_jsonl_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_claude_response(response: str) -> str:
    """
    Reuse same trimming logic as developer.py, but for reviewer we want text-only.
    We'll still keep a small sanitizer: strip code fences if any appear.
    """
    if response is None:
        return ""
    if "```" in response:
        # keep only text before fenced code or remove fences
        # simplest: strip fences entirely
        response = response.replace("```python", "").replace("```", "")
    return response.strip()


def infer_task_id_from_path(path: str):
    m = re.search(r"task_(\d+)", os.path.basename(path))
    return m.group(1) if m else None


def resolve_bias_file(task_id: str, bias_base_path: str) -> str:
    """
    bias_base_path can be:
      - a directory that contains task_<id>_bias_info.jsonl
      - a single file path (for one task)
    """
    if os.path.isdir(bias_base_path):
        return os.path.join(bias_base_path, f"task_{task_id}_bias_info.jsonl")
    return bias_base_path


# --- Bias check (you will implement later) ---
def check_bias(bias_obj) -> bool:
    """
    TODO: Replace with your real logic.

    Current conservative heuristic:
    - if bias_obj has truthy keys: has_bias / biased / bias => True
    - if bias_obj has non-empty list keys: issues / flags => True
    - else False
    """
    if not isinstance(bias_obj, dict):
        return False

    for k in ("has_bias", "biased", "bias"):
        if k in bias_obj and bool(bias_obj[k]):
            return True

    for k in ("issues", "flags"):
        if k in bias_obj and isinstance(bias_obj[k], list) and len(bias_obj[k]) > 0:
            return True

    return False


# --- LLM call (same multi-model switch style as developer.py) ---
def review_conversation(style, qs, temp, model_name) -> str:
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        txt = response.choices[0].message.content
        return (txt or "").strip()

    elif model_name == "llama":
        response = llama_client.chat.completions.create(
            model="CODELLAMA/CODELLAMA-70B-INSTRUCT-HF",
            temperature=temp,
            messages=[
                {"role": "system", "content": style},
                {"role": "user", "content": qs},
            ],
        )
        txt = response.choices[0].message.content
        return process_claude_response(txt)

    elif model_name == "bison":
        parameters = {
            "temperature": temp,
            "max_output_tokens": 256,
        }
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=style)
        response = chat.send_message(qs, **parameters)
        txt = response.text
        return (txt or "").strip()

    elif model_name == "claude":
        # keep same pause pattern as developer.py to avoid rate limits
        time.sleep(20)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=256,
            temperature=temp,
            system=style,
            messages=[{"role": "user", "content": [{"type": "text", "text": qs}]}],
        )
        txt = response.content[0].text
        return process_claude_response(txt)

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")


# --- Prompt styles (same dict pattern as developer.py) ---
prompt_styles = {
    "gpt": {
        "agent_reviewer": (
            "You are a fairness-focused code reviewer. "
            "Given a prompt, generated code, and bias_info, output ONLY brief, exact, actionable instructions "
            "to remove bias. Do NOT output code. Do NOT add extra text."
        )
    },
    "llama": {
        "agent_reviewer": (
            "You are a fairness-focused code reviewer. "
            "Output ONLY brief, exact instructions to remove bias (no code, no extra text)."
        )
    },
    "bison": {
        "agent_reviewer": (
            "You are a fairness-focused code reviewer. "
            "Return only short instructions to remove bias. No code."
        )
    },
    "claude": {
        "agent_reviewer": (
            "You are a fairness-focused code reviewer. "
            "Output ONLY brief, exact instructions to remove bias. Do not output code."
        )
    },
}


def generate_reviews_from_bias_info(
    prompts_file_path: str,
    output_dir: str,
    bias_info_base_path: str,
    iterations: int,
    temperature: float,
    style: str,
    model_name: str,
):
    """
    For each task in prompts_file_path:
      - read generated code from output_dir/task_<id>_generated_code.jsonl
      - read bias info from bias_info_base_path/task_<id>_bias_info.jsonl (or direct file)
      - write review lines to output_dir/task_<id>_review.jsonl

    Each line corresponds to an iteration index.
    """
    for json_obj in read_jsonl_file(prompts_file_path):
        task_id = str(json_obj.get("task_id", "default"))
        prompt = json_obj.get("prompt", "")

        if not prompt:
            continue

        gen_code_path = os.path.join(output_dir, f"task_{task_id}_generated_code.jsonl")
        if not os.path.exists(gen_code_path):
            print(f"[WARN] Missing generated code file: {gen_code_path}")
            continue

        bias_file_path = resolve_bias_file(task_id, bias_info_base_path)
        if not os.path.exists(bias_file_path):
            print(f"[WARN] Missing bias info file: {bias_file_path}")
            continue

        # Read code + bias lines
        code_lines = list(read_jsonl_file(gen_code_path))
        bias_lines = list(read_jsonl_file(bias_file_path))

        # Output review file (same dir, different file name)
        review_path = os.path.join(output_dir, f"task_{task_id}_review.jsonl")
        os.makedirs(os.path.dirname(review_path), exist_ok=True)

        with open(review_path, "w", encoding="utf-8") as out_f:
            # align by iteration index
            n = min(iterations, len(code_lines), len(bias_lines))
            for i in range(n):
                code_obj = code_lines[i] if isinstance(code_lines[i], dict) else {"generated_code": str(code_lines[i])}
                bias_obj = bias_lines[i] if isinstance(bias_lines[i], dict) else {"bias_info": bias_lines[i]}

                if not check_bias(bias_obj):
                    json.dump({"review": "pass"}, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    continue

                # Build concise user query
                qs = (
                    "PROMPT:\n"
                    f"{prompt}\n\n"
                    "GENERATED_CODE:\n"
                    f"{code_obj.get('generated_code','')}\n\n"
                    "BIAS_INFO:\n"
                    f"{json.dumps(bias_obj, ensure_ascii=False)}\n\n"
                    "Return ONLY brief instructions to remove the bias (no code)."
                )

                instruction = review_conversation(style, qs, temperature, model_name)
                json.dump({"review": instruction}, out_f, ensure_ascii=False)
                out_f.write("\n")

        print(f"[OK] Wrote reviews: {review_path}")


if __name__ == "__main__":
    """
    Mirrors developer.py arg style, with ONE EXTRA arg at the end:

    python reviewer.py <prompts_jsonl> <output_dir> <num_samples> [temperature] [prompt_style] <model_name> <bias_info_base_path>

    - prompts_jsonl: same input you used for developer.py
    - output_dir: same output dir where task_<id>_generated_code.jsonl exists
    - num_samples: iterations (same as developer)
    - bias_info_base_path: directory or file path for bias info
    """
    prompts_jsonl_path = sys.argv[1]
    output_base_dir = sys.argv[2]
    num_samples = int(sys.argv[3])

    TEMPERATURE = 1.0 if len(sys.argv) < 5 else float(sys.argv[4])
    PROMPT_STYLE = "agent_reviewer" if len(sys.argv) < 6 else sys.argv[5]
    MODEL_NAME = sys.argv[6]
    BIAS_INFO_BASE_PATH = sys.argv[7]

    print("prompts_jsonl_path", prompts_jsonl_path)
    print("output_base_dir", output_base_dir)
    print("num_samples", num_samples)
    print("TEMPERATURE", TEMPERATURE)
    print("PROMPT_STYLE", PROMPT_STYLE)
    print("MODEL_NAME", MODEL_NAME)
    print("BIAS_INFO_BASE_PATH", BIAS_INFO_BASE_PATH)

    os.makedirs(output_base_dir, exist_ok=True)

    generate_reviews_from_bias_info(
        prompts_file_path=prompts_jsonl_path,
        output_dir=output_base_dir,
        bias_info_base_path=BIAS_INFO_BASE_PATH,
        iterations=num_samples,
        temperature=TEMPERATURE,
        style=prompt_styles[MODEL_NAME][PROMPT_STYLE],
        model_name=MODEL_NAME,
    )
