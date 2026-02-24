import csv
import json
import os.path
import sys

# Base part of the file path without the numeric suffix and extension
model_dir = sys.argv[1]
test_start = int(sys.argv[2])
test_count = int(sys.argv[3])
agent = sys.argv[4]

base_dir = os.path.abspath(f"{model_dir}/test_result/{agent}")

# Initialize a list to hold the results for each file
all_file_results = []

# Loop through the file numbers
for i in range(test_start, test_count):  # Assuming the files are numbered from test_start to test_count
    file_name = f"summary_output_suite_task_{i}.csv"
    file_path = os.path.join(base_dir, "summary_bias", file_name)
    attribute_bias_counts = {}  # Reset for each file
    attribute_counts = {}  # To track the count of each attribute

    # Read data from the current CSV file
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            attribute = row[1]

            # Skip processing if the attribute name is "Attribute"
            if attribute == "Attribute":
                continue

            bias_exist_values = row[-1].split(', ')

            # Increment the count for the attribute
            attribute_counts[attribute] = attribute_counts.get(attribute, 0) + 1

            # Initialize the attribute in the bias dictionary if not already there
            if attribute not in attribute_bias_counts:
                attribute_bias_counts[attribute] = {}

            # Increment the count for each bias exist value
            for bias_value in bias_exist_values:
                value = bias_value.split(':')[0].strip('"')
                attribute_bias_counts[attribute][value] = attribute_bias_counts[attribute].get(value, 0) + 1

    # Append the results for this file to the all_file_results list
    all_file_results.append({
        "File Index": i,
        "Results": [{"Attribute": attr, "Count": attribute_counts[attr], "Bias Exist": bias} for attr, bias in attribute_bias_counts.items()]
    })

# Write the results to a JSON file
output_file_path = os.path.join(base_dir, 'bias_leaning_summary.json')
with open(output_file_path, 'w') as json_file:
    json.dump(all_file_results, json_file, indent=4)

print("Data from each CSV file has been processed and saved separately in bias_leaning_summary.json, "
      "including attribute counts and excluding the 'Attribute' named attributes.")
