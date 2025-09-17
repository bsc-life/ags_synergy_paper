# this small script compares two JSON files
# Main use is in order to explain differences in two specific instances
# For debugging and further understand this model, as well as how different parameters affect
# the growth curves in the simulation

import json, csv
from collections import OrderedDict

def compare_json_files(file1, file2, output_csv):
    # Load JSON data from files
    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)

    # Calculate differences
    differences = {}
    for key in data1.keys():
        if key in data2:
            diff = abs(float(data1[key]) - float(data2[key]))
            differences[key] = diff

    # Sort differences from most to least
    sorted_differences = OrderedDict(sorted(differences.items(), key=lambda x: x[1], reverse=True))

    # Print ranked differences with values from both files
    print("Ranked differences (most to least):")
    print(f"{'Key':<20} {'File 1':<10} {'File 2':<10} {'Difference':<10}")
    print("-" * 50)

    # Prepare data for CSV
    csv_data = [["Key", "File 1", "File 2", "Difference"]]

    for key, diff in sorted_differences.items():
        value1 = float(data1[key])
        value2 = float(data2[key])
        print(f"{key:<20} {value1:<10.4f} {value2:<10.4f} {diff:<10.4f}")
        csv_data.append([key, f"{value1:.4f}", f"{value2:.4f}", f"{diff:.4f}"])

    # Write to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(csv_data)

    print(f"\nResults have been saved to {output_csv}")


# Example usage
compare_json_files(
    file1='experiments/CMA-03_09_2024-01:31:52-10p_postdrug_combined_PI3K/instance_1_0_4/sim_summary.json', 
    file2='experiments/CMA-03_09_2024-01:31:52-10p_postdrug_combined_PI3K/instance_1_27_2/sim_summary.json',
    output_csv="results/json_comparisons/comparison_10p_PI3K.csv"
    )