import pandas as pd

jigsaw_df = pd.read_csv('../processed/jigsaw_cleaned.csv')
rutoxic_df = pd.read_csv('../processed/rutoxic_cleaned.csv')

combined_df = pd.concat([jigsaw_df, rutoxic_df], ignore_index=True)

combined_df.to_csv('../processed/combined_dataset.csv', index=False)

print("Datasets merged and saved as 'combined_dataset.csv'")