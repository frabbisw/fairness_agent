if [ "$#" -le 5 ]; then
    echo "Usage: bash gen_and_exp.sh [sampling] [temperature] [prompt_style] [data_path] [model_dir] [model_name]"
    exit
fi

CURRENT_DIR=$(pwd)

SAMPLING=$1
TEMPERATURE=$2
PROMPT_STYLE=$3
DATA_PATH=$4
MODEL_DIR=$5
MODEL_NAME=$6
TEST_COUNT=$7

echo "SAMPLING:" "$SAMPLING"
echo "TEMPERATURE:" "$TEMPERATURE"
echo "PROMPT_STYLE:" "$PROMPT_STYLE"
echo "DATA_PATH:" "$DATA_PATH"
echo "MODEL_DIR:" "$MODEL_DIR"
echo "MODEL_NAME:" "$MODEL_NAME"
echo "TEST_COUNT:" "$TEST_COUNT"
echo "-------------------"

echo developer.py "$DATA_PATH" "$MODEL_DIR"/response "$SAMPLING" "$TEMPERATURE" "$PROMPT_STYLE" "$MODEL_NAME"
echo parse_bias_info.py "$MODEL_DIR"/test_result_developer/log_files "$MODEL_DIR"/test_result_developer/bias_info_files "$SAMPLING"
echo summary_result.py "$MODEL_DIR"
echo count_bias.py "$MODEL_DIR"
echo count_bias_leaning.py "$MODEL_DIR"
echo "===================="


# Delete the previous result files
rm -rf "$MODEL_DIR""/test_result_developer"

#generate and save responses from model
cd "$CURRENT_DIR""/../generate_code" || exit
# python developer.py "$DATA_PATH" "$MODEL_DIR"/response "$SAMPLING" "$TEMPERATURE" "$PROMPT_STYLE" "$MODEL_NAME"

#run test suits
cd "$CURRENT_DIR""/../fairness_test/test_suites/" || exit

BASE_DIR="$MODEL_DIR""/response"
LOG_DIR="$MODEL_DIR""/test_result_developer/log_files"
REPORT_BASE_DIR="$MODEL_DIR""/test_result_developer/inconsistency_files"

cp config_template.py config.py
sed -i "s|##PATH##TO##RESPONSE##|$BASE_DIR|g" config.py
sed -i "s|##PATH##TO##LOG##FILES##|$LOG_DIR|g" config.py
sed -i "s|##PATH##TO##INCONSISTENCY##FILES##|$REPORT_BASE_DIR|g" config.py

pytest test_suite_0.py test_suite_1.py test_suite_2.py

#parse bias summary from log files
cd .. || exit
echo "developer parse_bias_info"
python parse_bias_info.py "$MODEL_DIR""/test_result_developer/log_files" "$MODEL_DIR""/test_result_developer/bias_info_files" "$SAMPLING"
echo "developer summary result"
python summary_result.py "$MODEL_DIR"
echo "developer count bias"
python count_bias.py "$MODEL_DIR" $TEST_COUNT
echo "developer count related"
python count_related.py "$MODEL_DIR" $TEST_COUNT
echo "developer count bias leaning"
python count_bias_leaning.py "$MODEL_DIR" $TEST_COUNT
