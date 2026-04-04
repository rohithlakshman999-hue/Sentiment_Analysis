import pandas as pd

df = pd.read_csv("data/tes.tsv", sep='\t')

# See column names (debug)
print("Columns:", df.columns)

# Take first column automatically
df = df.iloc[:, 0:1]

# Rename column
df.columns = ['review']

# Save as CSV
df.to_csv("data/reviews.csv", index=False)

print("Converted successfully ✅")