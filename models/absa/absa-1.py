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
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize_function, batched=True)
dev_dataset   = dev_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# === 4. Load Model ===
num_labels = len(train_df["labels"].unique())
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# === 5. Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# === 6. Training Arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=50,
    load_best_model_at_end=False,
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
results = trainer.evaluate(test_dataset)
print("Test Results:", results)






