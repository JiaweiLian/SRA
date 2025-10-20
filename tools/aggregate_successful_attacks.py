import os
import json
import argparse
from pathlib import Path

def aggregate_successful_attacks(base_dir, output_file):
    """
    Aggregates successful jailbreak attempts from multiple JSON result files.

    Args:
        base_dir (str): The base directory containing the LLM model folders.
                       e.g., 'results/SRA/transfer_attack'
        output_file (str): The path to save the aggregated results JSON file.
    """
    successful_cases = {}
    
    # Get all subdirectories in the base directory, which are the LLM model folders
    try:
        model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        print(f"Found {len(model_dirs)} model directories.")
    except FileNotFoundError:
        print(f"Error: Base directory not found at '{base_dir}'")
        return

    # Use the model_name argument to aggregate results for different models
    model_name = aggregate_successful_attacks.model_name if hasattr(aggregate_successful_attacks, 'model_name') else "gpt-3.5-turbo-1106"
    # Append model_name to output_file before .json
    output_path = Path(output_file)
    if output_path.suffix == ".json":
        output_file = str(output_path.with_name(f"{output_path.stem}_{model_name}.json"))

    # Iterate through each model's result directory
    for model_dir in model_dirs:
        # Construct the path to the specific results file
        json_path = Path(base_dir) / model_dir / model_name / "results" / f"{model_name}.json"
        
        if not json_path.exists():
            print(f"Warning: Results file not found for model '{model_dir}' at {json_path}")
            continue
            
        print(f"Processing file: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {json_path}")
                continue

        # Iterate through each behavior (attack scenario) in the JSON file
        for behavior, results in data.items():
            # If this behavior has already been successfully recorded, skip it
            if behavior in successful_cases:
                continue

            # Check if there are any results for this behavior
            if not results:
                continue

            # We only care about the first result entry, as per the file structure
            result_item = results[0]

            # Check if the attack was successful ("label": 1)
            if result_item.get("label") == 1:
                print(f"  [SUCCESS] Found successful case for behavior: '{behavior}' in model '{model_dir}'")
                successful_cases[behavior] = result_item

    # Save the aggregated results to the output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(successful_cases, f, indent=4, ensure_ascii=False)
    print(f"\nAggregation complete.")
    print(f"Found a total of {len(successful_cases)} unique successful attack cases.")
    print(f"Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate successful jailbreak attempts from multiple result files.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="results/SRA/transfer_attack",
        help="The base directory containing the LLM model folders."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/SRA/transfer_attack/aggregated_successful_attacks.json",
        help="The path to save the aggregated results JSON file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-3.5-turbo-1106",
        help="The model name to aggregate results for (e.g., gpt-3.5-turbo-1106, gpt-4-1106-preview, etc.)."
    )
    args = parser.parse_args()

    # Pass model_name as an attribute to the function for compatibility
    aggregate_successful_attacks.model_name = args.model_name
    aggregate_successful_attacks(args.base_dir, args.output_file)

if __name__ == "__main__":
    main()
