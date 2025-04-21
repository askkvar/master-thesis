import pandas as pd

# Load datasets
df1 = pd.read_pickle("data/processed/bunker_test_set_tokenized.pkl")
df2 = pd.read_pickle("data/processed/100_reviews_for_manual_annotation.pkl")

print("Dataset 1 shape:", df1.shape)
print("Dataset 2 shape:", df2.shape)

# Concatenate
merged_df = pd.concat([df1, df2], ignore_index=True)

# Drop duplicates based only on clean_text (or any subset of columns that makes sense)
merged_df = merged_df.drop_duplicates(subset=["clean_text"])

# Save the result
merged_df.to_pickle("data/processed/bunker_reviews_test_set.pkl")

print("Merged dataset saved successfully.")
