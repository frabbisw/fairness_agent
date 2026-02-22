import csv
from collections import defaultdict
import importlib
import glob  # Import glob to list CSV files
import sys
import os

sys.path.append(os.path.abspath('../fairness_test/test_suites'))

# Function to dynamically import attributes from a specified module
def dynamic_import(module_name, attribute_names):
    module = importlib.import_module(module_name)
    return [getattr(module, attr) for attr in attribute_names]


def calculate_total_test_cases(demographic_data_, attribute_):
    num_combinations = 1
    for key, values in demographic_data_.items():
        if key != attribute_:
            num_combinations *= len(values)
    return num_combinations


def parse_detail_and_determine_bias(detail_string):
    biases = {}
    entries = detail_string.split(', ')
    for entry in entries:
        key_value = entry.split(': ')
        if len(key_value) == 2:
            key, value = key_value
            biases[key] = value == 'True'
    return biases


def analyze_bias_direction(details):
    overall_biases = defaultdict(int)
    for detail in details:
        biases = parse_detail_and_determine_bias(detail)
        for key, is_biased in biases.items():
            if is_biased:
                overall_biases[key] += 1
    return dict(overall_biases)


def read_and_analyze_csv(file_path, demographic_data_):
    data = defaultdict(lambda: {'details': []})
    with open(file_path, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            attribute = row['Attribute']
            detail = row['Detail']
            data[attribute]['details'].append(detail)

    bias_summary = {}
    for attribute in SENSITIVE_ATTRIBUTES:
        total_cases = calculate_total_test_cases(demographic_data_, attribute)
        inconsistent_cases = len(data[attribute]['details'])
        inconsistency_ratio = inconsistent_cases / total_cases if total_cases else 0
        bias_direction = analyze_bias_direction(data[attribute]['details'])

        bias_summary[attribute] = {
            'total_cases': total_cases,
            'inconsistent_cases': inconsistent_cases,
            'inconsistency_ratio': inconsistency_ratio,
            'bias_exist': bias_direction
        }
    return bias_summary


# Function to list all CSV file paths in a given directory
def list_csv_files(directory_path):
    return glob.glob(f"{directory_path}/*.csv")


# Function to write summaries to a CSV file
def write_summaries_to_csv(summaries, output_file_path):
    """Writes bias analysis summaries to a CSV file, excluding entries without bias."""
    # Extract the directory path from the output file path
    directory = os.path.dirname(output_file_path)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)
    headers = ['File Path', 'Attribute', 'Total Cases', 'Inconsistent Cases', 'Inconsistency Ratio', 'Bias Exist']
    with open(output_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        for file_path, summary in summaries.items():
            for attribute, stats in summary.items():
                # Check if bias exists; if not, skip writing this row
                if not stats['bias_exist']:
                    continue

                bias_exist = ', '.join(f"{key}: {value}" for key, value in stats['bias_exist'].items())
                writer.writerow({
                    'File Path': file_path,
                    'Attribute': attribute,
                    'Total Cases': stats['total_cases'],
                    'Inconsistent Cases': stats['inconsistent_cases'],
                    'Inconsistency Ratio': f"{stats['inconsistency_ratio']:.2%}",
                    'Bias Exist': bias_exist
                })

model_dir = sys.argv[1]
test_count = int(sys.argv[2])
agent = sys.argv[3]

base_dir = os.path.abspath(f"{model_dir}/test_result/{agent}")

for number in range(test_count):
    # Dynamically construct the module name for the current test suite
    module_name = f'test_suites.test_suite_{number}'

    # Try to dynamically import DEMOGRAPHIC_DATA and SENSITIVE_ATTRIBUTES from the current test suite
    try:
        DEMOGRAPHIC_DATA, SENSITIVE_ATTRIBUTES = dynamic_import(module_name,
                                                                ['DEMOGRAPHIC_DATA', 'SENSITIVE_ATTRIBUTES'])
    except ImportError as e:
        print(f"Skipping {module_name} due to import error: {e}")
        continue

    # Dynamically construct the directory path for the current task
    directory_path = os.path.join(base_dir, f"inconsistency_files/task_{number}")

    # List all CSV files in the specified directory for the current task
    csv_file_paths = list_csv_files(directory_path)

    # Initialize a dictionary to store summaries for each file
    summaries = {}

    # Iterate over each file path, analyze it, and record the summary for the current task
    for file_path in csv_file_paths:
        summary = read_and_analyze_csv(file_path, DEMOGRAPHIC_DATA)
        summaries[file_path] = summary  # Record the summary with the file path as the key

    # Specify the output CSV file path, including the test suite and task number
    output_csv_file = f'summary_bias/summary_output_suite_task_{number}.csv'

    # Write the summaries to the specified CSV file for the current test suite and task
    write_summaries_to_csv(summaries, os.path.join(base_dir, output_csv_file))

    print(f"Summaries for test suite and task {number} have been written to {output_csv_file}")

