import os
import glob
import pandas as pd

# Folder paths
RAW_FOLDER = "data/raw"
GOOGLE_FOLDER = os.path.join(RAW_FOLDER, "google")
PROCESSED_FOLDER = "data/processed"

def count_csv_rows(file_path: str) -> int:
    """Read a CSV file and return the number of rows."""
    df = pd.read_csv(file_path)
    return len(df)

def count_pkl_rows(file_path: str) -> int:
    """Read a Pickle file and return the number of rows."""
    df = pd.read_pickle(file_path)
    return len(df)

def print_top_level_raw_counts():
    """
    Print row counts for CSV/PKL files in the raw folder
    (excluding the 'google' subfolder).
    """
    print("=== RAW DATASETS (Top-Level) ===")
    
    # List files in raw folder, ignoring the 'google' subfolder
    items = os.listdir(RAW_FOLDER)
    files = []
    for item in items:
        full_path = os.path.join(RAW_FOLDER, item)
        # Skip subfolders (including google)
        if os.path.isdir(full_path):
            continue
        # Only consider CSV or PKL
        if item.endswith(".csv") or item.endswith(".pkl"):
            files.append(full_path)
    
    # Sort for consistent display
    files.sort()
    
    if not files:
        print("No top-level raw files found.")
        return
    
    for f in files:
        file_name = os.path.basename(f)
        if f.endswith(".csv"):
            rows = count_csv_rows(f)
        else:  # pkl
            rows = count_pkl_rows(f)
        print(f"{file_name}: {rows} rows")

def print_google_combined_counts():
    """
    Sum all rows from CSV files in the google subfolder
    and treat them as one dataset.
    """
    print("\n=== RAW DATASETS (Google Subfolder) ===")
    csv_files = glob.glob(os.path.join(GOOGLE_FOLDER, "*.csv"))
    
    if not csv_files:
        print("No google CSV files found.")
        return
    
    total_rows = 0
    for f in csv_files:
        total_rows += count_csv_rows(f)
    
    print(f"google (combined): {total_rows} rows")

def print_processed_counts():
    """
    Print row counts for each CSV/PKL file in the processed folder.
    """
    print("\n=== PROCESSED DATASETS ===")
    patterns = [
        os.path.join(PROCESSED_FOLDER, "*.csv"),
        os.path.join(PROCESSED_FOLDER, "*.pkl")
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern))
    all_files.sort()
    
    if not all_files:
        print("No processed files found.")
        return
    
    for f in all_files:
        file_name = os.path.basename(f)
        if f.endswith(".csv"):
            rows = count_csv_rows(f)
        elif f.endswith(".pkl"):
            rows = count_pkl_rows(f)
        else:
            continue
        print(f"{file_name}: {rows} rows")

def main():
    print_top_level_raw_counts()
    if os.path.isdir(GOOGLE_FOLDER):
        print_google_combined_counts()
    print_processed_counts()

if __name__ == "__main__":
    main()
