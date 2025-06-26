import os
import json
import re


def aggregate_compression_results(root_dir: str, output_filename: str = "TACO.json") -> dict:
    """
    Aggregate mean_stat.json results from multiple compression experiments in a directory.

    Args:
        root_dir (str): Root directory containing multiple lambda_* subdirectories
        output_filename (str): Output JSON file name (saved in root_dir)

    Returns:
        dict: Statistics data for all lambda experiments
    """
    total_data = {}
    lambda_pattern = re.compile(r'lambda_([\d.]+)')

    for dir_name in os.listdir(root_dir):
        match = lambda_pattern.search(dir_name)
        if match:
            lambda_value = float(match.group(1))
            json_path = os.path.join(root_dir, dir_name, "mean_stat.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as file:
                    data = json.load(file)
                total_data[lambda_value] = data
            else:
                print(f"Warning: {json_path} does not exist")

    total_data = dict(sorted(total_data.items()))
    output_path = os.path.join(root_dir, output_filename)
    with open(output_path, 'w') as file:
        json.dump(total_data, file, indent=4)

    return total_data


if __name__ == "__main__":
    base_dir = "compression_kodak_results"
    aggregate_compression_results(base_dir, output_filename="kodak_taco.json")
