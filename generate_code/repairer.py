# generate_code/repairer.py
import json
import os
import sys
import time

from openai import OpenAI
from dotenv import load_dotenv
from google.cloud import datastore
from vertexai.preview.language_models import CodeGenerationModel  # keep same style as developer.py
from vertexai.language_models import CodeChatModel
import anthropic

load_dotenv()

# --- Clients (same style as developer.py) ---
gpt_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

llama_client = OpenAI(
    api_key=os.environ.get("TOGETHER_API_KEY"),
    base_url='https://api.together.xyz/v1',
)

# set google client using: https://cloud.google.com/vertex-ai/docs/start/client-libraries
# google_client = datastore.Client()

anthropic_client = anthropic.Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def process_claude_response(response):
    """
    Same behavior as developer.py:
    - ensure output starts from 'def '
    - truncate after blank line / code fence
    - wrap as python fenced block
    """
    if response is None:
        return "# NO CODE GENERATED"
    if "def " not in response:
        return "# NO CODE GENERATED"
    if not response.startswith("def "):
        response = response[response.find("def "):]
    if "\n\n" in response:
        response = response[:response.find("\n\n")]
    if "```" in response:
        response = response[:response.find("```")]
    response = "```python\n" + response + "\n```\n"
    return response


def repair_conversation(style, qs, temp, model_name):
    """
    Same multi-model switch style as developer.py.
    """
    if model_name == "gpt":
        response = gpt_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=temp,
            messages=[{"role": "system", "content": style}, {"role": "user", "content": qs}],
        )
        code = response.choices[0].message.content

    elif model_name == "llama":
        response = llama_client.chat.completions.create(
            model="CODELLAMA/CODELLAMA-70B-INSTRUCT-HF",
            temperature=temp,
            messages=[{"role": "system", "content": style}, {"role": "user", "content": qs}],
        )
        code = response.choices[0].message.content
        code = process_claude_response(code)

    elif model_name == "bison":
        parameters = {
            "temperature": temp,
            "max_output_tokens": 512,
        }
        code_chat_model = CodeChatModel.from_pretrained("codechat-bison@002")
        chat = code_chat_model.start_chat(context=style)
        response = chat.send_message(qs, **parameters)
        code = response.text
        if code.startswith(" "):
            code = code[1:]

    elif model_name == "claude":
        time.sleep(20)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=512,
            temperature=temp,
            system=style,
            messages=[{"role": "user", "content": [{"type": "text", "text": qs}]}]
        )
        code = response.content[0].text
        code = process_claude_response(code)

    else:
        raise ValueError("Invalid model name. Choose between 'gpt', 'llama', 'bison', 'claude'.")

    return code


# Prompt styles (same dict pattern)
prompt_styles = {
    "gpt": {        
        "agent": '''You are a code repairer. Apply the given JSON edits to CURRENT_METHOD exactly. 
        "old" strings must be matched verbatim; perform operations in order. 
        Output ONLY the final repaired method code (signature+body). 
        No class, no markdown, no extra text. Do not change the method signature (including self).'''
    }
}


def generate_repaired_code(
    prompts_file_path: str,
    src_gc_dir: str,
    src_review_dir: str,
    target_repair_dir: str,
    iterations: int,
    temperature: float,
    style: str,
    model_name: str,
    test_start: int,
    test_end: int
):
    """
    For each task in prompts_file_path:
      - reads developer outputs from src_gc_dir/task_<id>_generated_code.jsonl
      - reads reviewer outputs from src_review_dir/task_<id>_review.jsonl
      - writes repaired outputs to target_repair_dir/task_<id>_repaired_code.jsonl

    Each line corresponds to an iteration index.
    """
    # for json_obj in read_jsonl_file(prompts_file_path):
    for index, json_obj in enumerate(islice(read_jsonl_file(prompts_file_path), test_start, test_end), start=test_start):
        print(f"Processing line {index}")
        print("-"*50)
        task_id = str(json_obj.get("task_id", "default"))
        prompt = json_obj.get("prompt", "")

        if not prompt:
            continue

        gen_code_path = os.path.join(src_gc_dir, f"task_{task_id}_generated_code.jsonl")
        review_path = os.path.join(src_review_dir, f"task_{task_id}_review.jsonl")

        if not os.path.exists(gen_code_path):
            print(f"[WARN] Missing generated code file: {gen_code_path}")
            continue
        if not os.path.exists(review_path):
            print(f"[WARN] Missing review file: {review_path}")
            continue

        code_lines = list(read_jsonl_file(gen_code_path))
        review_lines = list(read_jsonl_file(review_path))

        out_path = os.path.join(target_repair_dir, f"task_{task_id}_generated_code.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as out_f:
            n = min(iterations, len(code_lines), len(review_lines))
            for i in range(n):
                code_obj = code_lines[i]
                rev_obj = review_lines[i]

                current_method = code_obj.get("generated_code", "")
                review_text = (rev_obj.get("review", "") or "").strip()

                # If no bias / pass => keep the original method
                if review_text.lower() == "pass":
                    json.dump({"generated_code": current_method}, out_f, ensure_ascii=False)
                    out_f.write("\n")
                    continue

                qs = (
                    "CURRENT_METHOD:\n"
                    f"{current_method}\n\n"
                    "EDITS_JSON:\n"
                    f"{review_text}\n"
                )

                repaired_code = repair_conversation(style, qs, temperature, model_name)
                json.dump({"generated_code": repaired_code}, out_f, ensure_ascii=False)
                out_f.write("\n")

        print(f"[OK] Wrote repaired code: {out_path}")


if __name__ == "__main__":
    """
    Same CLI style as your reviewer.py (explicit args):

    python repair.py <prompts_jsonl> <src_gc_dir> <src_review_dir> <target_repair_dir> <num_samples> <temperature> <prompt_style> <model_name>
    """
    print("starting repairer agent")
    print("=" * 50)

    prompts_jsonl_path = sys.argv[1]
    src_gc_base_dir = sys.argv[2]
    src_review_base_dir = sys.argv[3]
    target_repair_base_dir = sys.argv[4]
    num_samples = int(sys.argv[5])
    TEMPERATURE = float(sys.argv[6])
    PROMPT_STYLE = sys.argv[7]
    MODEL_NAME = sys.argv[8]
    TEST_START = sys.argv[9]
    TEST_END = sys.argv[10]
    

    print("prompts_jsonl_path", prompts_jsonl_path)
    print("src_gc_base_dir", src_gc_base_dir)
    print("src_review_base_dir", src_review_base_dir)
    print("target_repair_base_dir", target_repair_base_dir)
    print("num_samples", num_samples)
    print("TEMPERATURE", TEMPERATURE)
    print("PROMPT_STYLE", PROMPT_STYLE)
    print("MODEL_NAME", MODEL_NAME)
    print("TEST_START", TEST_START)
    print("TEST_END", TEST_END)

    os.makedirs(target_repair_base_dir, exist_ok=True)

    generate_repaired_code(
        prompts_file_path=prompts_jsonl_path,
        src_gc_dir=src_gc_base_dir,
        src_review_dir=src_review_base_dir,
        target_repair_dir=target_repair_base_dir,
        iterations=num_samples,
        temperature=TEMPERATURE,
        style=prompt_styles[MODEL_NAME][PROMPT_STYLE],
        model_name=MODEL_NAME,
        test_start=int(TEST_START),
        test_end=int(TEST_END)
    )
