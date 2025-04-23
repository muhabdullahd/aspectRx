"""
Train and evaluate ABSA model on CADEC healthcare dataset.
This script adapts the ABSA model to work with healthcare domain reviews.
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.MetricCallback import MetricsCallback

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Configuration
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "../../results/cadec-absa"
LOG_DIR = "../../training_plots/cadec"
METRICS_FILE = os.path.join(OUTPUT_DIR, "metrics_cadec.json")
SAVED_MODEL_PATH = os.path.join(OUTPUT_DIR, "cadec-absa-model")
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 6
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def load_cadec_data():
    """Load the CADEC ABSA datasets."""
    print("Loading CADEC ABSA datasets...")
    
    train_df = pd.read_csv("../../Dataset/cadec_absa_train.tsv", delimiter="\t")
    val_df = pd.read_csv("../../Dataset/cadec_absa_val.tsv", delimiter="\t")
    test_df = pd.read_csv("../../Dataset/cadec_absa_test.tsv", delimiter="\t")
    
    # Convert tokens column from string representation to actual lists
    for df in [train_df, val_df, test_df]:
        df['tokens'] = df['tokens'].apply(lambda x: x.strip('[]').replace("'", "").split(', '))
    
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def prepare_dataset_for_hf(df):
    """Prepare dataset for Hugging Face Transformers."""
    # Create text input by combining relevant fields
    df["text_input"] = df.apply(
        lambda row: f"{' '.join(row['tokens'])} [SEP] {row['absa2']} [SEP] {str(row['absa3'])}",
        axis=1
    )
    
    # For binary sentiment classification (positive vs non-positive)
    # Change from numeric sentiment (0,1,2) to binary (0=non-positive, 1=positive)
    df["label"] = df["absa1"].apply(lambda x: 1 if x == 2 else 0)
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[["text_input", "label"]])
    return dataset

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset using the provided tokenizer."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text_input"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["text_input"])
    return tokenized_dataset

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    return {
        "accuracy": accuracy,
        "f1": f1
    }

def train_cadec_model(train_dataset, val_dataset):
    """Train CADEC ABSA model and return trainer and metrics"""
    
    # Model setup
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2,  # Binary classification (positive/negative)
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="../../results/cadec-absa",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="../../logs/cadec-absa",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Create list to store evaluation metrics manually
    eval_metrics = []
    
    # Define compute_metrics function
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        
        # Store metrics
        current_metrics = {
            "accuracy": float(accuracy),
            "f1": float(f1),
            "step": len(eval_metrics) + 1
        }
        eval_metrics.append(current_metrics)
        
        return {"accuracy": accuracy, "f1": f1}
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Extract training metrics
    train_metrics = train_result.metrics
    
    # Create pandas DataFrame from collected metrics
    metrics_df = pd.DataFrame(eval_metrics)
    
    # Save model
    model_save_path = "../../results/cadec-absa/cadec-absa-model"
    trainer.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Print final training metrics
    print(f"Training metrics: {train_metrics}")
    
    return trainer, metrics_df

def evaluate_model(trainer, tokenized_test):
    """Evaluate the trained model on the test set."""
    print("Evaluating model on test set...")
    results = trainer.evaluate(tokenized_test)
    
    print(f"Test accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1 score: {results['eval_f1']:.4f}")
    
    # Get predictions
    predictions = trainer.predict(tokenized_test)
    preds = predictions.predictions.argmax(-1)
    labels = predictions.label_ids
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Positive', 'Positive'],
                yticklabels=['Non-Positive', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - CADEC ABSA')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'confusion_matrix.png'))
    
    return results

def plot_training_metrics(metrics_df):
    """Plot training metrics."""
    plt.figure(figsize=(10, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(metrics_df['step'], metrics_df['accuracy'], 'o-', label='Accuracy')
    plt.xlabel('Evaluation Step')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.plot(metrics_df['step'], metrics_df['f1'], 'o-', label='F1 Score')
    plt.xlabel('Evaluation Step')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_metrics.png'))
    print(f"Training metrics plots saved to {os.path.join(LOG_DIR, 'training_metrics.png')}")

def main():
    # Load the CADEC ABSA datasets
    train_df, val_df, test_df = load_cadec_data()
    
    # Prepare datasets for Hugging Face
    train_dataset = prepare_dataset_for_hf(train_df)
    val_dataset = prepare_dataset_for_hf(val_df)
    test_dataset = prepare_dataset_for_hf(test_df)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)
    
    # Train the model
    trainer, metrics_df = train_cadec_model(tokenized_train, tokenized_val)
    
    # Plot training metrics
    plot_training_metrics(metrics_df)
    
    # Evaluate the model on test set
    test_results = evaluate_model(trainer, tokenized_test)
    
    print("Training and evaluation complete!")
    print(f"Check results in {OUTPUT_DIR} and {LOG_DIR}")

if __name__ == "__main__":
    main()
