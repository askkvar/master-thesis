import os
import glob
import pandas as pd

def main():
    processed_folder = "data/processed"
    pkl_files = glob.glob(os.path.join(processed_folder, "*.pkl"))
    
    if not pkl_files:
        print("No .pkl files found in", processed_folder)
        return
    
    for pkl_file in pkl_files:
        df = pd.read_pickle(pkl_file)
        csv_file = pkl_file.replace(".pkl", ".csv")  # same path, .csv extension
        df.to_csv(csv_file, index=False)
        print(f"Converted {pkl_file} -> {csv_file}")

if __name__ == "__main__":
    main()
