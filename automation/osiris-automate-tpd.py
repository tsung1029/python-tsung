import subprocess
import csv
import os
from tabulate import tabulate

import re

def update_a0_value(source_file, output_file, new_a0):
    """
    Reads from source_file, replaces the 'a0' value, 
    and writes the result to output_file.
    """
    try:
        # 1. Read the template content
        with open(source_file, 'r') as f:
            content = f.read()

        # 2. Use Regex to find a0=... and replace the numeric value
        # Pattern handles spaces, decimals, and scientific notation
        pattern = r"(a0\s*=\s*)([0-9.eE+-]+)"
        replacement = rf"\g<1>{new_a0}"
        
        new_content = re.sub(pattern, replacement, content)

        # 3. Write the modified content to the new output file
        with open(output_file, 'w') as f:
            f.write(new_content)
            
        print(f"Created {output_file} with a0 = {new_a0}")
        
    except FileNotFoundError:
        print(f"Error: The source file '{source_file}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Example usage:
# update_a0_value("master_config.txt", "run_1.input", 0.0075)

def process_output(output_dir):
    """
    Extracts data from the directory. 
    Replace this with your actual data extraction logic.
    """
    # Placeholder: Assuming we find some values based on the directory
    x_val = 10.5  # Example float
    y_val = 20.2  # Example float
    return x_val, y_val

def run_automation(iterations, filename="filename"):
    results = []
    headers = ["Run_ID", "Directory", "X_Value", "Y_Value"]
    
    for i in range(iterations):
        dir_name = f"output_run_{i}"
        
        try:
            print(f"Executing a.out for {dir_name}...")
            # Run the command: ./a.out output_run_i
            subprocess.run(["./a.out", dir_name], check=True)
            
            # Call your function to get the data pair
            x, y = process_output(dir_name)
            results.append([i, dir_name, x, y])
            
        except subprocess.CalledProcessError:
            print(f"Skipping run {i} due to a.out error.")
            continue

    # --- Writing to the file ---
    
    # Option A: Save as CSV (Best for data analysis/Excel)
    with open(f"{filename}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(results)
    
    # Option B: Save as a Pretty Text Table (Best for reading)
    with open(filename, "w") as f:
        f.write(tabulate(results, headers=headers, tablefmt="grid"))

    print(f"\nSuccess! Data saved to '{filename}' and '{filename}.csv'")

if __name__ == "__main__":
    run_automation(iterations=5)