import json
import argparse

def compare_json_files(file1_path, file2_path):
    """
    Compares two JSON files and prints the differences in their top-level keys.

    Args:
        file1_path (str): The path to the first JSON file.
        file2_path (str): The path to the second JSON file.
    """
    try:
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
        
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)

        keys1 = set(data1.keys())
        keys2 = set(data2.keys())

        diff_in_file2 = keys2 - keys1
        diff_in_file1 = keys1 - keys2

        if not diff_in_file1 and not diff_in_file2:
            print("The two JSON files have the same set of keys.")
            return

        if diff_in_file2:
            print(f"\nKeys present in '{file2_path}' but not in '{file1_path}':")
            for key in diff_in_file2:
                print(f"  - Key: '{key}'")
                print(f"    Value: {data2[key]}")

        if diff_in_file1:
            print(f"\nKeys present in '{file1_path}' but not in '{file2_path}':")
            for key in diff_in_file1:
                print(f"  - Key: '{key}'")
                print(f"    Value: {data1[key]}")

    except FileNotFoundError as e:
        print(f"Error: {e}. Please provide valid file paths.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}. Please ensure files are valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two JSON files to find differing keys.")
    parser.add_argument("--file1", help="Path to the first JSON file.", default="/root/HarmBench/results/FewShot/qwen3_4b_thinking/test_cases/test_cases.json")
    parser.add_argument("--file2", help="Path to the second JSON file.", default="/root/HarmBench/results/FewShot/deepseek_r1_distill_llama_70b/test_cases/test_cases.json")

    args = parser.parse_args()

    print(f"Comparing '{args.file1}' and '{args.file2}'...")
    compare_json_files(args.file1, args.file2)
