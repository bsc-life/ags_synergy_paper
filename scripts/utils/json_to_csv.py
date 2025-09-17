import json
import csv

def convert_json_to_csv(json_file, csv_file):
    # Read JSON data from input file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract headers from the JSON data (assuming all records have the same structure)
    headers = list(data[0].keys())
    
    # Write data to CSV file
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        
        # Write header row
        writer.writeheader()
        
        # Write each record as a row in the CSV file
        writer.writerows(data)

# Example usage: Convert 'input.json' to 'output.csv'


if __name__ == "__main__":

    input_json_file = '/home/oth/BSC/NORD3/EMEWS/drug_synergy_emews/data/deap_AGS_2D_14p_MEK.json'
    output_csv_file = '/home/oth/BSC/NORD3/EMEWS/drug_synergy_emews/data/deap_AGS_2D_14p_MEK.csv'

    convert_json_to_csv(input_json_file, output_csv_file)
