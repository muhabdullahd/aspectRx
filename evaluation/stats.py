import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

# === Set dataset folder path ===
dataset_dir = "Dataset"  # Update if needed
train_file = os.path.join(dataset_dir, "cadec_absa_train.tsv")
dev_file   = os.path.join(dataset_dir, "cadec_absa_val.tsv")
test_file  = os.path.join(dataset_dir, "cadec_absa_test.tsv")

# === Load datasets ===
train_df = pd.read_csv(train_file, sep="\t")
dev_df   = pd.read_csv(dev_file, sep="\t")
test_df  = pd.read_csv(test_file, sep="\t")

# Add split labels
train_df["split"] = "train"
dev_df["split"] = "dev"
test_df["split"] = "test"

# Combine into one DataFrame
df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

# Convert tokens from string to list
df["tokens"] = df["tokens"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Add token length column
df["token_length"] = df["tokens"].apply(len)

# === Create output directory for plots ===
output_dir = "dataset_stats"
os.makedirs(output_dir, exist_ok=True)

# === 1. Label distribution (sentiment) ===
plt.figure(figsize=(8, 5))
sns.countplot(x="absa1", data=df, palette="Set2")
plt.title("Sentiment Label Distribution")
plt.xlabel("Label (absa1)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "label_distribution.png"))
plt.close()

# === 2. Token length distribution ===
plt.figure(figsize=(8, 5))
sns.histplot(df["token_length"], bins=30, kde=True)
plt.title("Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "token_length_distribution.png"))
plt.close()

# === 3. Aspect frequency ===
plt.figure(figsize=(10, 6))
top_aspects = df["absa2"].value_counts().nlargest(10)
sns.barplot(y=top_aspects.index, x=top_aspects.values, palette="coolwarm")
plt.title("Top 10 Most Common Aspects")
plt.xlabel("Count")
plt.ylabel("Aspect Term")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "aspect_distribution.png"))
plt.close()

# === 4. Dataset split sizes ===
plt.figure(figsize=(6, 4))
sns.countplot(x="split", data=df, palette="pastel")
plt.title("Dataset Split Sizes")
plt.xlabel("Split")
plt.ylabel("Samples")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "split_sizes.png"))
plt.close()

print("✅ All dataset visualizations saved to the 'dataset_stats' folder.")
