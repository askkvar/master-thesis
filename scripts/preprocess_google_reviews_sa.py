import pandas as pd
import re
import glob
import os

# Define input and output paths
input_folder = "data/raw/google/"
output_file = "data/processed/processed_bunker_sentiment.pkl"

# Get all CSV files
csv_files = glob.glob(os.path.join(input_folder, "google_*.csv"))

# Function to preprocess text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase (align with movie reviews)
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove usernames
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Initialize list to store reviews
all_reviews = []

for file in csv_files:
    df = pd.read_csv(file)
    
    # Extract relevant columns
    if "text" in df.columns:
        df = df[['text']]
        df['bunker'] = os.path.basename(file).replace("google_", "").replace(".csv", "")  # Track bunker

        # Clean text
        df['clean_text'] = df['text'].apply(clean_text)

        # Assign unique index
        df['index'] = df.index.astype(str) + "_" + df['bunker']

        all_reviews.append(df[['index', 'clean_text', 'bunker']])

# Merge all data and save
final_df = pd.concat(all_reviews, ignore_index=True)
final_df.set_index("index", inplace=True)  # Set unique index

final_df.to_pickle(output_file)

print(f"Preprocessing complete. Sentiment dataset saved to {output_file}")
