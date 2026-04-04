import bz2
import pandas as pd

reviews = []

with bz2.open("data/test.ft.txt.bz2", "rt", encoding="utf-8") as file:
    for i, line in enumerate(file):
        # Split label and text
        parts = line.strip().split(' ', 1)

        if len(parts) == 2:
            review = parts[1]
            reviews.append(review)

        # Limit to 10k (IMPORTANT)
        if i >= 10000:
            break

# Convert to DataFrame
df = pd.DataFrame(reviews, columns=['review'])

# Save as CSV
df.to_csv("data/reviews.csv", index=False)

print("Converted successfully ✅")