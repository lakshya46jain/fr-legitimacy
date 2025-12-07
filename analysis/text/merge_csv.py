import pandas as pd

# Load both CSVs
df1 = pd.read_csv("./snapshot_texts_lakshya.csv")
df2 = pd.read_csv("./snapshot_texts_rishita.csv")

# Append/concatenate them
merged = pd.concat([df1, df2], ignore_index=True)

# Save to a new file
merged.to_csv("./snapshot_texts.csv", index=False)