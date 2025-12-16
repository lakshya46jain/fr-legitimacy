import pandas as pd

# Load topics file
df = pd.read_csv("../analysis/topics/topics.csv")

# Count by decade
decade_counts = df['decade'].value_counts().sort_index()

# Percentage distribution
decade_percent = (decade_counts / decade_counts.sum()) * 100

summary = pd.DataFrame({
    "documents": decade_counts,
    "percentage": decade_percent.round(2)
})

print(summary)