import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# File paths
data_path = "data/processed/processed_bunker_sentiment.pkl"
model_path = "notebooks/bunker_multi_class/final_model"

# Load data
df = pd.read_pickle(data_path)

# Remove bunkers with < 100 reviews
bunker_counts = df["bunker_name"].value_counts()
valid_bunkers = bunker_counts[bunker_counts >= 100].index
df = df[df["bunker_name"].isin(valid_bunkers)].copy()

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Run on GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict_sentiment_batch(token_batch):
    input_ids = torch.tensor([x["input_ids"] for x in token_batch]).to(device)
    attention_mask = torch.tensor([x["attention_mask"] for x in token_batch]).to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu().tolist()

# Predict in batches
batch_size = 32
tokens = df["tokens"].tolist()
predictions = []

for i in tqdm(range(0, len(tokens), batch_size)):
    batch = tokens[i:i+batch_size]
    preds = predict_sentiment_batch(batch)
    predictions.extend(preds)

# Map predictions to scores
df["predicted_label"] = predictions
label_to_score = {0: -1, 1: 0, 2: 1}
df["sentiment_score"] = df["predicted_label"].map(label_to_score)

# Aggregate sentiment score per bunker
bunker_scores = df.groupby("bunker_name")["sentiment_score"].mean().reset_index()
bunker_scores = bunker_scores.sort_values(by="sentiment_score", ascending=False)

# Save or print
print(bunker_scores)

# Optionally save to file
bunker_scores.to_csv("data/processed/bunker_sentiment_ranking.csv", index=False)
