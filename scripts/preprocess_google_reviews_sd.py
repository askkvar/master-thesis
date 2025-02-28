import pandas as pd
import re
import glob
import os
from transformers import BertTokenizer

# Load BERT tokenizer (using bert-base-uncased to match your sample tokens)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define input and output paths
input_folder = "data/raw/google/"
output_file = "data/processed/processed_bunker_sarcasm.pkl"

# Get all CSV files
csv_files = glob.glob(os.path.join(input_folder, "google_*.csv"))

# Cleaning function: remove HTML breaks, URLs, usernames, and extra spaces,
# but preserve uppercase letters, emojis, and repeated characters.
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"http\S+|www.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Function to select the appropriate review text:
# if the review is non-English and a translated text exists, use it; otherwise, use the original.
def select_review_text(row):
    original_lang = row.get("originalLanguage", None)
    text = row.get("text", "")
    text_translated = row.get("textTranslated", "")
    if original_lang is not None and original_lang != "en" and pd.notna(text_translated) and text_translated.strip() != "":
        return text_translated
    return text

# Tokenization function using the BERT tokenizer
def tokenize_text(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze().tolist(),
        "attention_mask": encoded["attention_mask"].squeeze().tolist()
    }

# List to store processed reviews from all CSV files
all_reviews = []

for file in csv_files:
    df = pd.read_csv(file)
    if "text" in df.columns:
        # Create the 'review' column using the appropriate text (translated if needed)
        df["review"] = df.apply(select_review_text, axis=1)
        # Clean the review text (keeping uppercase and exaggerations)
        df["clean_text"] = df["review"].apply(clean_text)
        # Filter out rows with empty cleaned text
        df = df[df["clean_text"] != ""]
        # Tokenize the cleaned text
        df["tokens"] = df["clean_text"].apply(tokenize_text)
        # Keep only the columns that match the desired format: review, clean_text, tokens.
        all_reviews.append(df[["review", "clean_text", "tokens"]])

if all_reviews:
    final_df = pd.concat(all_reviews, ignore_index=True)
    final_df.to_pickle(output_file)
    print(f"Preprocessing complete. Bunker reviews dataset saved to {output_file}")
else:
    print("No reviews found.")
