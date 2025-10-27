import pandas as pd

# Load your dataset
df = pd.read_csv('../data/arxiv_data.csv')

# Create summaries from titles
df['summaries'] = df['title']
df['abstracts'] = df['abstract']

# Keep only needed columns
df_clean = df[['abstracts', 'summaries']].copy()

# Remove rows with missing data
df_clean = df_clean.dropna()

# Save new dataset
df_clean.to_csv('../data/arxiv_data.csv', index=False)

print(f"✅ Dataset prepared!")
print(f"   Total samples: {len(df_clean)}")
print(f"\n📊 Sample:")
print(f"Abstract: {df_clean.iloc[0]['abstracts'][:200]}...")
print(f"Summary: {df_clean.iloc[0]['summaries']}")