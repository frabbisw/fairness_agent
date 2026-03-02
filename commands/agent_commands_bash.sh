#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -le 5 ]; then
  echo "Usage: bash gen_and_exp.sh [sampling] [temperature] [prompt_style] [data_path] [model_dir] [model_name] [test_start] [test_count]"
  exit 1
fi

CURRENT_DIR="$(pwd)"

SAMPLING="$1"
TEMPERATURE="$2"
PROMPT_STYLE="$3"
DATA_PATH="$4"
MODEL_DIR="$5"
MODEL_NAME="$6"
TEST_START="${7:-0}"
TEST_COUNT="${8:-0}"

# output filename for prompt modifier (same dir as DATA_PATH)
MODIFIED_PROMPTS_FILENAME="prompts_modified.jsonl"
DATA_DIR="$(dirname "$(realpath "$DATA_PATH")")"
MODIFIED_DATA_PATH="$DATA_DIR/$MODIFIED_PROMPTS_FILENAME"

echo "SAMPLING: $SAMPLING"
echo "TEMPERATURE: $TEMPERATURE"
echo "PROMPT_STYLE: $PROMPT_STYLE"
echo "DATA_PATH: $DATA_PATH"
echo "MODIFIED_DATA_PATH: $MODIFIED_DATA_PATH"
echo "MODEL_DIR: $MODEL_DIR"
echo "MODEL_NAME: $MODEL_NAME"
echo "TEST_START: $TEST_START"
echo "TEST_COUNT: $TEST_COUNT"
echo "-------------------"

# -------------------------
# Helpers
# -------------------------

run_generator() {
  # $1 = script (prompt_modifier.py / developer.py / reviewer.py / repairer.py)
  # remaining args are passed as-is
  local script="$1"; shift
  cd "$CURRENT_DIR/../generate_code" || exit 1
  python "$script" "$@"
}

prepare_pytest_config_and_run() {
  # $1 = response_base_dir
  # $2 = log_dir
  # $3 = report_base_dir
  local response_dir="$1"
  local log_dir="$2"
  local report_dir="$3"

  cd "$CURRENT_DIR/../fairness_test/test_suites/" || exit 1

  cp config_template.py config.py
  sed -i "s|##PATH##TO##RESPONSE##|$response_dir|g" config.py
  sed -i "s|##PATH##TO##LOG##FILES##|$log_dir|g" config.py
  sed -i "s|##PATH##TO##INCONSISTENCY##FILES##|$report_dir|g" config.py

  pytest
}

run_postprocess() {
  # $1 = agent_name (developer/repairer)
  local agent="$1"

  cd "$CURRENT_DIR/../fairness_test/" || exit 1

  echo "$agent parse_bias_info"
  python parse_bias_info.py \
    "$MODEL_DIR/test_result/$agent/log_files" \
    "$MODEL_DIR/test_result/$agent/bias_info_files" \
    "$SAMPLING"

  echo "$agent summary result"
  python summary_result.py "$MODEL_DIR" "$TEST_START" "$TEST_COUNT" "$agent"

  echo "$agent count bias"
  python count_bias.py "$MODEL_DIR" "$TEST_START" "$TEST_COUNT" "$agent"

  echo "$agent count related"
  python count_related.py "$MODEL_DIR" "$TEST_START" "$TEST_COUNT" "$agent"

  echo "$agent count bias leaning"
  python count_bias_leaning.py "$MODEL_DIR" "$TEST_START" "$TEST_COUNT" "$agent"
}

run_test_phase() {
  # $1 = agent_name (developer/repairer)
  # $2 = response_dir (MODEL_DIR/response/<agent>)
  local agent="$1"
  local response_dir="$2"

  rm -rf "$MODEL_DIR/test_result/$agent"

  local log_dir="$MODEL_DIR/test_result/$agent/log_files"
  local report_dir="$MODEL_DIR/test_result/$agent/inconsistency_files"

  prepare_pytest_config_and_run "$response_dir" "$log_dir" "$report_dir"
  run_postprocess "$agent"
}

# -------------------------
# Pipeline
# -------------------------

echo "prompt_modifier.py $DATA_PATH -> $MODIFIED_DATA_PATH"
echo "developer.py $MODIFIED_DATA_PATH $MODEL_DIR/response/developer $SAMPLING $TEMPERATURE $PROMPT_STYLE $MODEL_NAME"
echo "parse_bias_info.py $MODEL_DIR/test_result/developer/log_files $MODEL_DIR/test_result/developer/bias_info_files $SAMPLING"
echo "summary_result.py $MODEL_DIR"
echo "count_bias.py $MODEL_DIR"
echo "count_bias_leaning.py $MODEL_DIR"
echo "===================="

# 0) Prompt modifier: rewrite prompts and save alongside input prompts
# NOTE: prompt_modifier.py expects positional args in the version I gave earlier.
# Provide CONTEXT_MESSAGE via env var (recommended) to avoid quoting issues:
#   export CONTEXT_MESSAGE="...."

run_generator prompt_modifier.py \
  --jsonl_input_file_path="$DATA_PATH" \
  --output_prompt_filename="prompts_modified.jsonl" \
  --num_samples=1 \
  --temperature="$TEMPERATURE" \
  --prompt_style="$PROMPT_STYLE" \
  --model_name="$MODEL_NAME" \
  --test_start="$TEST_START" \
  --test_end="$TEST_COUNT"

# 1) Developer: generate (use modified prompts)
run_generator developer.py \
  --jsonl_input_file_path="$MODIFIED_DATA_PATH" \
  --output_base_dir="$MODEL_DIR/response/developer" \
  --num_samples="$SAMPLING" \
  --temperature="$TEMPERATURE" \
  --prompt_style="$PROMPT_STYLE" \
  --model_name="$MODEL_NAME" \
  --test_start="$TEST_START" \
  --test_end="$TEST_COUNT"

# 2) Developer: test + parse + summarize
run_test_phase "developer" "$MODEL_DIR/response/developer"

# 3) Reviewer: generate (use modified prompts for consistency)
run_generator reviewer.py \
  --prompts_jsonl_path="$MODIFIED_DATA_PATH" \
  --src_gc_base_dir="$MODEL_DIR/response/developer" \
  --target_review_base_dir="$MODEL_DIR/response/reviewer" \
  --num_samples="$SAMPLING" \
  --temperature="$TEMPERATURE" \
  --prompt_style="$PROMPT_STYLE" \
  --model_name="$MODEL_NAME" \
  --bias_info_base_path="$MODEL_DIR/test_result/developer/bias_info_files" \
  --test_start="$TEST_START" \
  --test_end="$TEST_COUNT"

# 4) Repairer: generate
run_generator repairer.py \
  --prompts_jsonl_path="$MODIFIED_DATA_PATH" \
  --src_gc_base_dir="$MODEL_DIR/response/developer" \
  --src_review_base_dir="$MODEL_DIR/response/reviewer" \
  --target_repair_base_dir="$MODEL_DIR/response/repairer" \
  --num_samples="$SAMPLING" \
  --temperature="$TEMPERATURE" \
  --prompt_style="$PROMPT_STYLE" \
  --model_name="$MODEL_NAME" \
  --test_start="$TEST_START" \
  --test_end="$TEST_COUNT"

# 5) Repairer: test + parse + summarize
run_test_phase "repairer" "$MODEL_DIR/response/repairer"
