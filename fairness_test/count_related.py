import json
import os.path
import sys


def count_bias_attributes(file_path):
    attribute_counts = {}
    total_objects = 0  # Total number of JSON objects in the file
    objects_with_bias = 0  # Count objects with at least one bias attribute

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            related_info = data.get('related_info', '')

            if related_info == "failed":
                continue

            total_objects += 1

            if related_info != "none" and related_info != "":
                attributes = [attr.strip() for attr in related_info.split(',') if attr.strip()]
                objects_with_bias += 1

                for attribute in attributes:
                    attribute_counts[attribute] = attribute_counts.get(attribute, 0) + 1

    # Calculate bias ratios
    bias_ratios = {attribute: (count / objects_with_bias) for attribute, count in
                   attribute_counts.items()} if objects_with_bias else {}
    general_bias_ratio = (objects_with_bias / total_objects) if total_objects else 0

    return attribute_counts, objects_with_bias, total_objects, bias_ratios, general_bias_ratio


# Initialize dictionaries to hold counts and results
all_results = {}

model_path = sys.argv[1]
test_count = int(sys.argv[2])

base_dir = os.path.abspath(f"{model_path}/test_result")

# Loop through the file numbers, starting from 0 to 342
for i in range(test_count):  # test_count files, starting from index 0
    file_name = f'related_info{i}.jsonl'
    file_path = os.path.join(base_dir, "bias_info_files", file_name)

    try:
        attribute_counts, objects_with_bias, total_objects, bias_ratios, general_bias_ratio = count_bias_attributes(
            file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        continue

    # Record the bias ratios with the file number as the key
    all_results[f'{i}'] = {
        'attribute_counts': attribute_counts,
        'objects_with_related': objects_with_bias,
        'total_objects': total_objects,
        # 'bias_ratios': bias_ratios,
        # 'general_bias_ratio': general_bias_ratio
    }

# Write the aggregated results to a single file
output_file_path = os.path.join(base_dir, 'aggregated_related_ratios_after.json')
with open(output_file_path, 'w') as output_file:
    json.dump(all_results, output_file, indent=4)

print(f"Aggregated related ratios, have been written to {output_file_path}.")
