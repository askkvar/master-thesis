import pandas as pd
from transformers import BertTokenizer

# Initialize the tokenizer (using bert-base-uncased for consistency)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the tokenization function
def tokenize_text(text):
    encoded = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,  # Adjust if needed (e.g., 128 or 256)
        return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].squeeze().tolist(),
        "attention_mask": encoded["attention_mask"].squeeze().tolist()
    }

# Load the Excel file (assumes it has columns 'clean_text' and 'manual_classification')
df = pd.read_excel("data/processed/bunker_test_for_manual_annotation2.xlsx")

# Tokenize the 'clean_text' column and add as a new column 'tokens'
df["tokens"] = df["clean_text"].apply(tokenize_text)

# Optionally, check the first few rows to ensure everything looks as expected
print(df.head())

# Save the DataFrame as a pickle file for later use
output_pickle = "data/processed/bunker_test_set_tokenized2.pkl"
df.to_pickle(output_pickle)
print(f"Tokenized test set saved to {output_pickle}")
