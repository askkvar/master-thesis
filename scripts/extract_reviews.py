import pandas as pd

# Load the datasets
classified_df = pd.read_pickle("data/processed/bunker_test_set_tokenized.pkl")
all_reviews_df = pd.read_pickle("data/processed/processed_bunker_sentiment.pkl")

# Exclude already classified reviews
remaining_reviews_df = all_reviews_df[~all_reviews_df['clean_text'].isin(classified_df['clean_text'])]

# Sample 100 random reviews
sampled_reviews_df = remaining_reviews_df.sample(n=100, random_state=42)

# Save to pickle for further processing
sampled_reviews_df.to_pickle("data/processed/processed_bunker_test_set2.pkl")

print("New 100 reviews saved to: data/processed/processed_bunker_test_set2.pkl")