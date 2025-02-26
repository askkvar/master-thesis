import pandas as pd


# df = pd.read_pickle("data/processed/processed_bunker_sentiment.pkl")
# df.to_csv("data/processed/processed_bunker_sentiment.csv", index=True)

# df = pd.read_pickle("data/processed/processed_bunker_sarcasm.pkl")
# df.to_csv("data/processed/processed_bunker_sarcasm.csv", index=True)

# df = pd.read_pickle("data/processed/processed_movie_reviews.pkl")
# df.to_csv("data/processed/processed_movie_reviews.csv", index=True)

df = pd.read_pickle("data/processed/processed_twitter_train.pkl")
df.to_csv("data/processed/processed_twitter_train.csv", index=True)