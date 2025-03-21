import os
import glob
import pandas as pd

def main():
    folder_path = "data/raw/google"
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        file_name = os.path.basename(csv_file)
        print(f"{file_name}: {len(df)} reviews")

if __name__ == "__main__":
    main()
