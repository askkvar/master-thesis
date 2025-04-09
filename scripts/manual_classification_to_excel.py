import pandas as pd

# Load your test set (pickle file)
df = pd.read_pickle("data/processed/700_more_reviews_for_ft.pkl")

# Create a new DataFrame with only the 'clean_text' column
# and add a new column for manual classification.
df_manual = df[['clean_text']].copy()
df_manual['manual_classification'] = ""  # This column will be filled manually

# Save the DataFrame to an Excel file
excel_file = "data/processed/700_reviews_for_manual_annotation.xlsx"
df_manual.to_excel(excel_file, index=False)

print(f"Excel file created: {excel_file}")
