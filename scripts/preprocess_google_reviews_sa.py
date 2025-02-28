import pandas as pd
import re
import glob
import os
import torch
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define input and output paths
input_folder = "data/raw/google/"
output_file = "data/processed/processed_bunker_sentiment.pkl"

# Get all CSV files
csv_files = glob.glob(os.path.join(input_folder, "google_*.csv"))

# Text cleaning function (matches the movie reviews cleaning)
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove HTML line breaks
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9.,!?']+", " ", text)  # Keep words, numbers, punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = re.sub(r"([.!?]){2,}", r"\1", text)  # Normalize repeated punctuation
    return text

# Function to select the appropriate review text: if non-English and a translated text exists, use it.
def select_review_text(row):
    original_lang = row.get("originalLanguage", None)
    text = row.get("text", "")
    text_translated = row.get("textTranslated", "")
    
    # If original language is not English and there's a non-empty translated version, use it.
    if original_lang is not None and original_lang != "en" and pd.notna(text_translated) and text_translated.strip() != "":
        return text_translated
    return text

# Tokenize text using BERT tokenizer (same as in the movie reviews dataset)
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
        
        # Clean the review text
        df["clean_text"] = df["review"].apply(clean_text)
        
        # Filter out rows with empty cleaned text
        df = df[df["clean_text"] != ""]
        
        # Tokenize the cleaned text
        df["tokens"] = df["clean_text"].apply(tokenize_text)
        
        # Keep only the columns that match the training dataset format: review, clean_text, tokens.
        all_reviews.append(df[["review", "clean_text", "tokens"]])

if all_reviews:
    final_df = pd.concat(all_reviews, ignore_index=True)
    final_df.to_pickle(output_file)
    print(f"Preprocessing complete. Bunker reviews dataset saved to {output_file}")
else:
    print("No reviews found.")
