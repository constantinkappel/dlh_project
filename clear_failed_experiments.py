# This deletes all experiments that have the "done" flag set to "false" in outputs/experiments.xlsx
import pandas as pd
from pathlib import Path
import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    # Load the experiments file
    experiments_file = Path("outputs/experiments.xlsx")
    experiments = pd.read_excel(experiments_file)

    # Get the failed experiments
    failed_experiments = experiments[experiments["done"] == False]
    deleted = 0
    if not args.simulate:
        # Delete the failed experiments
        for i, row in failed_experiments.iterrows():
            run = Path(row["run"])
            os.system(f"rm -rf {run}")
            deleted += 1
        print(f"{deleted} experiments deleted.")
        # re-run `python parse_experiments.py` to update the experiments file
        os.system("python parse_experiments.py")
    else:
        print("Simulating deletion of failed experiments.")
        print(f"Found {len(failed_experiments)} failed experiments.")
        for i, row in failed_experiments.iterrows():
            run = Path(row["run"])
            print(f"Deleting {row['run']} (--- SIMULATED ---)")
