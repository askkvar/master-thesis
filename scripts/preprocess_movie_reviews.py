import pandas as pd
import re
import torch
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Load dataset (modify the path to your dataset file)
file_path = "data/raw/imdb_reviews.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Select relevant columns: 'review' (review text) and 'sentiment' (label)
df = df[['review', 'sentiment']]

# Convert sentiment into numerical labels (1 = Positive, 0 = Negative)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Text cleaning function
def clean_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"<br\s*/?>", " ", text)  # Remove HTML line breaks
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9.,!?']+", " ", text)  # Keep words, numbers, punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    text = re.sub(r"([.!?]){2,}", r"\1", text)  # Normalize repeated punctuation
    return text

# Apply text cleaning
df['clean_text'] = df['review'].apply(clean_text)

# Tokenize text using BERT tokenizer
def tokenize_text(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze().tolist(),  # Keep only input_ids
        "attention_mask": encoded["attention_mask"].squeeze().tolist()  # Keep only attention_mask
    }

# Apply tokenization
df["tokens"] = df["clean_text"].apply(tokenize_text)

# Save processed dataset
df.to_pickle("data/processed/processed_movie_reviews.pkl")  # Save as a pickle file for later use
