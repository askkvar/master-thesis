import pandas as pd

# Load the datasets
classified_df = pd.read_pickle("data/processed/bunker_test_set_tokenized.pkl")
classified_df2 = pd.read_pickle("data/processed/bunker_test_set_tokenized2.pkl")
classified_df3 = pd.read_pickle("data/processed/700_more_reviews_for_ft.pkl")
all_reviews_df = pd.read_pickle("data/processed/processed_bunker_sentiment.pkl")

# Exclude already classified reviews
remaining_reviews_df = all_reviews_df[
    ~all_reviews_df['clean_text'].isin(classified_df['clean_text']) &
    ~all_reviews_df['clean_text'].isin(classified_df2['clean_text']) &
    ~all_reviews_df['clean_text'].isin(classified_df3['clean_text'])
    ]

# Sample x random reviews
sampled_reviews_df = remaining_reviews_df.sample(n=100, random_state=42)

# Save to pickle for further processing
sampled_reviews_df.to_pickle("data/processed/100_more_reviews_for_testing.pkl")

print("New 100 reviews saved to: data/processed/700_more_reviews_for_testing.pkl")