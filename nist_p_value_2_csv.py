import csv
import os
import re


def parse_p_value_file(file_path):
    """Parse the p-value information file and extract data"""
    with open(file_path, "r") as f:
        content = f.read()

    # Split by file sections
    sections = content.strip().split("\n\n")

    data = []
    for section in sections:
        lines = section.strip().split("\n")
        if not lines:
            continue

        # Extract model name from first line
        model_line = lines[0]
        model_match = re.search(r"\./text/(.+)\.txt", model_line)
        if not model_match:
            continue

        model_name = model_match.group(1)

        # Parse p-values from remaining lines
        for line in lines[1:]:
            if line.strip() and line.startswith("(") and line.endswith(")"):
                # Extract test name and p-value
                match = re.match(r"\(([^,]+),([^)]+)\)", line)
                if match:
                    test_name = match.group(1)
                    p_value = float(match.group(2))
                    data.append({"Model": model_name, "Test": test_name, "P_Value": p_value})

    return data


def convert_to_csv(data, output_file):
    """Convert parsed data to CSV format"""
    # Get unique models and tests for proper ordering
    models = sorted(list(set(item["Model"] for item in data)))
    tests = sorted(list(set(item["Test"] for item in data)))

    # Create CSV with models as columns and tests as rows
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        header = ["Test"] + models
        writer.writerow(header)

        # Write data rows
        for test in tests:
            row = [test]
            for model in models:
                # Find p-value for this test and model
                p_value = None
                for item in data:
                    if item["Model"] == model and item["Test"] == test:
                        p_value = item["P_Value"]
                        break
                row.append(p_value if p_value is not None else "")
            writer.writerow(row)


def main():
    input_file = "p_value_info.txt"
    output_file = "nist_p_values.csv"

    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    # Parse the data
    data = parse_p_value_file(input_file)

    if not data:
        print("No data found in the file!")
        return

    # Convert to CSV
    convert_to_csv(data, output_file)

    print(f"Successfully converted {input_file} to {output_file}")
    print(f"Found {len(data)} data points from {len(set(item['Model'] for item in data))} models")
    print(f"Tests included: {', '.join(sorted(set(item['Test'] for item in data)))}")


if __name__ == "__main__":
    main()
