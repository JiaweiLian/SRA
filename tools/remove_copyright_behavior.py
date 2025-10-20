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
    
path = 'results/GCG/deepseek_r1_distill_llama_8b/results/deepseek_r1_distill_llama_8b.json'
    
data = read_json_file(path)
data_removed_copyright = {}

# First, filter out items with 'Copyright' category
for item in data:
    if 'copyright' != data[item][0]['FunctionalCategory']:
        data_removed_copyright[item] = data[item]

# Create a new file path with "_removed_copyright" added
base_path, ext = os.path.splitext(path)
new_path = base_path + "_removed_copyright" + ext

# Save the filtered data to the new JSON file once after filtering
with open(new_path, 'w') as f:
    json.dump(data_removed_copyright, f, indent=4)
    
print(f"Filtered data saved to {new_path}")
