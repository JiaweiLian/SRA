import os
import json
import shutil
import argparse

def filter_templated_folders(folder_path):
    """
    Filter out folders where method_config.json has use_chat_template set to true.
    
    Args:
        folder_path (str): Path to the parent folder containing subfolders to check
    """
    # Ensure the input path exists and is a directory
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: {folder_path} does not exist or is not a directory")
        return
    
    # Get list of subdirectories
    subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    
    for subdir in subdirs:
        subdir_path = os.path.join(folder_path, subdir)
        config_file = os.path.join(subdir_path, "method_config.json")
        
        # Check if method_config.json exists
        if os.path.exists(config_file):
            try:
                # Read the JSON file
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if use_chat_template is true
                if config.get("use_chat_template", False) is False:
                    print(f"Deleting {subdir_path} (use_chat_template=True)")
                    shutil.rmtree(subdir_path)
                else:
                    print(f"Keeping {subdir_path} (use_chat_template=False or not set)")
            
            except json.JSONDecodeError:
                print(f"Error: {config_file} is not a valid JSON file. Keeping the directory.")
            except Exception as e:
                print(f"Error processing {subdir_path}: {str(e)}. Keeping the directory.")
        else:
            print(f"No method_config.json found in {subdir_path}. Keeping the directory.")

def main():
    parser = argparse.ArgumentParser(description='Filter out folders with use_chat_template=true in method_config.json')
    parser.add_argument('--folder_path', default='/home/jiawei/HarmBench/results/SRA/llama2_13b/test_cases/test_cases_individual_behaviors', 
                        help='Path to the directory containing subfolders to check')
    args = parser.parse_args()
    
    filter_templated_folders(args.folder_path)

if __name__ == "__main__":
    main()