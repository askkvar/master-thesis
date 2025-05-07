import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("data/processed/processed_movie_reviews.csv")

# Focus on the 'review' column and drop any NaNs
reviews = df['review'].dropna()

# Tokenize each review by splitting on whitespace
review_lengths = reviews.apply(lambda x: len(x.split()))

# Calculate statistics
average_length = review_lengths.mean()
median_length = review_lengths.median()
min_length = review_lengths.min()
max_length = review_lengths.max()

# Get shortest and longest reviews
shortest_review = reviews[review_lengths.idxmin()]
longest_review = reviews[review_lengths.idxmax()]
# Find a review with approximately 20 words (±2 words)
short_review_near_20 = reviews[(review_lengths >= 18) & (review_lengths <= 22)].sample(1).values[0]

print("\nExample review (~20 words):")
print(short_review_near_20)


# Print results
print(f"Average review length (in words): {average_length:.2f}")
print(f"Median review length (in words): {median_length}")
print(f"Shortest review length: {min_length} words")
print(f"Longest review length: {max_length} words\n")


print("\nShortest review:")
print(shortest_review)

print("\nLongest review:")
print(longest_review)

# --- PLOT DISTRIBUTION ---
plt.figure(figsize=(12, 7))
plt.hist(review_lengths, bins=100, color='skyblue', edgecolor='black', alpha=0.7)

# Annotate the longest review
plt.annotate('Longest Review',
             xy=(max_length, 1),
             xytext=(max_length - 600, 800),
             arrowprops=dict(arrowstyle='->', color='black', lw=2),
             fontsize=14,
             fontweight='bold',
             ha='right')

# Bold median line and label
plt.axvline(median_length, color='red', linestyle='dashed', linewidth=3,
            label=rf"$\mathbf{{Median:\ {int(median_length)}\ words}}$")

# Title and axis labels
plt.title('Distribution of Review Lengths (in Words)', fontsize=18, fontweight='bold')
plt.xlabel('Number of Words per Review', fontsize=16, fontweight='bold')
plt.ylabel('Number of Reviews', fontsize=16, fontweight='bold')

# Bold tick labels
plt.xticks(fontsize=13, fontweight='bold')
plt.yticks(fontsize=13, fontweight='bold')

# Legend with bold label
plt.legend(fontsize=13, loc='upper right', frameon=True)

# Grid and layout
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
