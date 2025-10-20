import csv
import json
import os

def read_json_file(file_path):
    """
    Read and parse a JSON file from the given path
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict/list: The parsed JSON data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None
    
def read_csv_file(file_path):
    """
    Read and parse a CSV file from the given path
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        list: The parsed CSV data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            behaviors = list(reader)
        return behaviors
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
results_path = 'results/SRA/llama2_7b_125000/test_cases/logs.json'
data_path = 'data/behavior_datasets/harmbench_behaviors_text_all.csv'
    
results = read_json_file(results_path)
data = read_csv_file(data_path)

# First, filter out items with 'Copyright' category
for item in results:
    for i in range(len(data)):
        if item == data[i]['BehaviorID']: # compare behavior ID
            results[item][0]['FunctionalCategory'] = data[i]['FunctionalCategory']
            results[item][0]['BehaviorID'] = data[i]['BehaviorID']
            results[item][0]['SemanticCategory'] = data[i]['SemanticCategory']
            results[item][0]['Behavior'] = data[i]['Behavior']
            results[item][0]['ContextString'] = data[i]['ContextString']
            break

# Create a new file path with "_supplemented" added
base_path, ext = os.path.splitext(results_path)
new_path = base_path + "_supplemented" + ext

# Save the filtered data to the new JSON file once after filtering
with open(new_path, 'w') as f:
    json.dump(results, f, indent=4)
    
print(f"Filtered data saved to {new_path}")
