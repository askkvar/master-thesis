import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, ClassLabel

# Load dataset from Hugging Face
dataset = load_dataset("jbeno/sentiment_merged")
label_names = ["negative", "neutral", "positive"]
dataset = dataset.cast_column("label", ClassLabel(names=label_names))

# Convert to pandas DataFrame (using the 'train' split)
df = dataset['train'].to_pandas()

# Use the 'sentence' column instead of 'review'
reviews = df['sentence'].dropna()
review_lengths = reviews.apply(lambda x: len(x.split()))

# Compute stats
average_length = review_lengths.mean()
median_length = review_lengths.median()
min_length = review_lengths.min()
max_length = review_lengths.max()

# Get example reviews
shortest_review = reviews[review_lengths.idxmin()]
longest_review = reviews[review_lengths.idxmax()]
short_review_near_20 = reviews[(review_lengths >= 18) & (review_lengths <= 22)].sample(1).values[0]

# Print results
print("\nExample review (~20 words):")
print(short_review_near_20)

print(f"\nAverage review length (in words): {average_length:.2f}")
print(f"Median review length (in words): {median_length}")
print(f"Shortest review length: {min_length} words")
print(f"Longest review length: {max_length} words")

print("\nShortest review:")
print(shortest_review)

print("\nLongest review:")
print(longest_review)

# --- PLOT DISTRIBUTION ---
plt.figure(figsize=(12, 7))
plt.hist(review_lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

# Get histogram data for dynamic placement
counts, bins = np.histogram(review_lengths, bins=100)
y_max = counts.max()

# Annotate the longest review with shorter arrow and better placement
plt.annotate('Longest Review',
             xy=(max_length, 1),                 # Arrow tip at actual max point
             xytext=(175, y_max * 0.15),          # Label between 150 and 200
             arrowprops=dict(arrowstyle='->', color='black', lw=2, shrinkA=0, shrinkB=5),
             fontsize=14,
             fontweight='bold',
             ha='center')


# Bold median line and label
plt.axvline(median_length, color='red', linestyle='dashed', linewidth=3,
            label=rf"$\mathbf{{Median:\ {int(median_length)}\ words}}$")

# Title and labels
plt.title('Distribution of Review Lengths (in Words)', fontsize=18, fontweight='bold')
plt.xlabel('Number of Words per Review', fontsize=16, fontweight='bold')
plt.ylabel('Number of Reviews', fontsize=16, fontweight='bold')

# Axis ticks
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

# Legend
plt.legend(fontsize=13, loc='upper right', frameon=True)

# Final layout
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
