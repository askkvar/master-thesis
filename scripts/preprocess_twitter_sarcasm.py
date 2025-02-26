import pandas as pd
import re
import torch
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# File paths (modify as needed)
train_file = "data/raw/twitter_reviews_train.csv"
test_file = "data/raw/twitter_reviews_test.csv"

# Load datasets
df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

# Select relevant columns
df_train = df_train[['tweets', 'class']]
df_test = df_test[['tweets', 'class']]

# Filter to keep only "regular" and "sarcastic" tweets
df_train = df_train[df_train['class'].isin(['regular', 'sarcastic'])]
df_test = df_test[df_test['class'].isin(['regular', 'sarcastic'])]

# Define preprocessing function
def clean_text(text):
    if pd.isna(text):  # Handle missing values
        return ""
    
    text = re.sub(r"http\S+|www.\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove usernames (@mentions)
    text = re.sub(r"#\w+", "", text)  # Remove hashtags (e.g., #sarcastic)
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text  # Keep uppercase, emojis, and punctuation

# Apply text cleaning
df_train['clean_text'] = df_train['tweets'].apply(clean_text)
df_test['clean_text'] = df_test['tweets'].apply(clean_text)

# Tokenize text using BERT tokenizer
def tokenize_text(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,  # Twitter data is short
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze().tolist(),  # Keep only input_ids
        "attention_mask": encoded["attention_mask"].squeeze().tolist()  # Keep only attention_mask
    }

# Apply tokenization
df_train['tokens'] = df_train['clean_text'].apply(tokenize_text)
df_test['tokens'] = df_test['clean_text'].apply(tokenize_text)

# Save processed datasets
df_train.to_pickle("data/processed/processed_twitter_train.pkl")
df_test.to_pickle("data/processed/processed_twitter_test.pkl")

print("Preprocessing complete. Processed files saved!")
