import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import sys
sys.path.append('../..')  # Add parent directory to path
from utils.MetricCallback import MetricsCallback
from utils.data_utils_absa import preprocess_for_absa

# === 1. Load and Preprocess Data ===
train_df = pd.read_csv("../../Dataset/cadec_absa_train.tsv", delimiter="\t")
dev_df   = pd.read_csv("../../Dataset/cadec_absa_val.tsv", delimiter="\t")
test_df  = pd.read_csv("../../Dataset/cadec_absa_test.tsv", delimiter="\t")

train_df, dev_df, test_df = preprocess_for_absa(train_df, dev_df, test_df)

# === 2. Convert to Hugging Face Datasets ===
train_dataset = Dataset.from_pandas(train_df)
dev_dataset   = Dataset.from_pandas(dev_df)
test_dataset  = Dataset.from_pandas(test_df)

# === 3. Tokenizer and Tokenization Function ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    # Using a longer max_length to accommodate the additional sentiment information
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=156)

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset   = dev_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

# Include original ABSA columns in the dataset format
columns_to_format = ["input_ids", "attention_mask", "labels"]
if "original_aspect" in train_dataset.column_names:
    columns_to_format.extend(["original_aspect", "original_sentiment"])

train_dataset.set_format("torch", columns=columns_to_format)
dev_dataset.set_format("torch", columns=columns_to_format)
test_dataset.set_format("torch", columns=columns_to_format)

# === 4. Load Model ===
num_labels = len(train_df["labels"].unique())
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# === 5. Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Calculate standard metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    
    # Return all metrics
    return {
        "accuracy": accuracy,
        "f1": f1,
        "combined_score": (accuracy + f1) / 2  # Simple combined metric
    }

# === 6. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Save at each epoch to track progress
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="f1",   # Use F1 score to determine the best model
    greater_is_better=True        # Higher F1 is better
)

# === 7. Trainer Setup ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback("results/metrics.json")]
)

# === 8. Train + Evaluate ===
trainer.train()
trainer.save_model("./saved_model")

# Evaluate on test dataset
results = trainer.evaluate(test_dataset)
print("Test Results:", results)

# === 9. Final Analysis with All ABSA Columns ===
# Perform predictions with the best model
test_preds = trainer.predict(test_dataset)
pred_labels = np.argmax(test_preds.predictions, axis=1)

# Print detailed analysis
print("\n=== Detailed Analysis ===")
print(f"Total test samples: {len(test_dataset)}")
print(f"Model accuracy: {results['eval_accuracy']:.4f}")
print(f"Model F1 score: {results['eval_f1']:.4f}")
print(f"Combined score: {results['eval_combined_score']:.4f}")
print("\nThe model has been trained using all three ABSA columns for enhanced sentiment analysis.")






