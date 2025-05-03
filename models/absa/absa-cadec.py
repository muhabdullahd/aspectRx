"""
Train and evaluate ABSA model on CADEC healthcare dataset.
This script adapts the ABSA model to work with healthcare domain reviews.
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
import re
import time # Add time import
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from nlpaug.augmenter.word import SynonymAug
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer, 
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
import json

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

# Custom Trainer class with weighted loss function
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Apply class weights to the loss
        if self.class_weights is not None:
            # Make sure labels and logits are on the same device as the weights
            device = self.class_weights.device
            logits = logits.to(device)
            labels = labels.to(device)
            
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            # Standard loss calculation
            loss = outputs.loss
            
        return (loss, outputs) if return_outputs else loss

def load_cadec_data():
    """Load the CADEC ABSA datasets."""
    print("Loading CADEC ABSA datasets...")
    # Use absolute path based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.abspath(os.path.join(script_dir, '../../Dataset'))
    train_path = os.path.join(dataset_dir, 'cadec_absa_train.tsv')
    val_path = os.path.join(dataset_dir, 'cadec_absa_val.tsv')
    test_path = os.path.join(dataset_dir, 'cadec_absa_test.tsv')

    train_df = pd.read_csv(train_path, delimiter="\t")
    val_df = pd.read_csv(val_path, delimiter="\t")
    test_df = pd.read_csv(test_path, delimiter="\t")

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
    
    # Add domain-specific features
    df = enhance_dataset_with_features(df)
    
    # Select features to include
    feature_cols = ["text_input", "label", "contains_side_effect", "contains_benefit", 
                   "high_severity", "contains_negation", "contains_comparison", "discontinued"]
    available_cols = [col for col in feature_cols if col in df.columns]
    
    # Convert to Hugging Face Dataset
    dataset = Dataset.from_pandas(df[available_cols])
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
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Standard metrics
    accuracy = accuracy_score(labels, preds)
    
    # Use macro-F1 or weighted-F1 for imbalanced data
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def train_cadec_model(trainer, metrics_callback): # Accept trainer and callback
    """Train CADEC ABSA model using the provided trainer."""
    
    print("Starting model training...")
    # Train the model
    train_result = trainer.train()
    
    # Extract training metrics
    train_metrics = train_result.metrics
    
    # Create pandas DataFrame from collected metrics via callback
    metrics_df = pd.DataFrame(metrics_callback.metrics_list) # Use callback data
    
    # Print final training metrics
    print(f"Training metrics: {train_metrics}")
    
    # Return only the metrics DataFrame (trainer is already available in main)
    return metrics_df 

def evaluate_model(trainer, tokenized_test, test_df, threshold=0.5): # Add test_df argument
    """Evaluate the trained model on the test set and save detailed predictions.""" # Updated docstring
    print("Evaluating model on test set...")
    
    start_time = time.time() # Start timing inference
    predictions = trainer.predict(tokenized_test)
    inference_time = time.time() - start_time # Calculate inference time
    print(f"Inference time: {inference_time:.2f} seconds")

    labels = predictions.label_ids
    
    # Apply custom threshold if not using default
    if threshold != 0.5:
        print(f"Using optimal threshold: {threshold:.4f}")
        preds = apply_threshold(predictions, threshold)
    else:
        preds = predictions.predictions.argmax(-1)
    
    # Calculate metrics with the optimal threshold
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)  # Standard F1 score (positive class)
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score (standard): {f1:.4f}")
    print(f"Test F1 score (macro): {f1_macro:.4f}")
    print(f"Test F1 score (weighted): {f1_weighted:.4f}")
    
    # Compare with default threshold results
    print("\nComparison with default threshold (0.5):")
    default_preds = predictions.predictions.argmax(-1)
    default_f1 = f1_score(labels, default_preds)
    default_f1_macro = f1_score(labels, default_preds, average='macro')
    print(f"Default threshold F1: {default_f1:.4f}")
    print(f"Default threshold Macro F1: {default_f1_macro:.4f}")
    print(f"Improvement: {(f1-default_f1)*100:.2f}% (standard F1), {(f1_macro-default_f1_macro)*100:.2f}% (macro F1)")
    
    # Create confusion matrix
    cm = confusion_matrix(labels, preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Positive', 'Positive'],
                yticklabels=['Non-Positive', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - CADEC ABSA (threshold={threshold:.2f})')
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'confusion_matrix_optimal.png'))
    
    # --- Save detailed predictions with aspect categories ---
    # Add print statements to check lengths
    print(f"DEBUG: Length of test_df passed to evaluate_model: {len(test_df)}")
    print(f"DEBUG: Length of prediction labels: {len(labels)}")
    
    # Ensure test_df has the same length as predictions
    if len(test_df) == len(labels):
        pred_df = pd.DataFrame({
            'true_label': labels,
            'predicted_label': preds,
            'aspect_category': test_df['absa3'].values # Use 'absa3' column for aspect category
        })
        
        # --- Construct Absolute Path ---
        script_dir = os.path.dirname(os.path.abspath(__file__))
        absolute_output_dir = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR)) # Make OUTPUT_DIR absolute
        pred_output_path = os.path.join(absolute_output_dir, 'test_predictions_with_aspects.csv') # Use absolute dir
        # --- End Absolute Path ---

        print(f"DEBUG: Attempting to save CSV to absolute path: {pred_output_path}") # UPDATED
        try: 
            # Ensure the directory exists before saving
            os.makedirs(absolute_output_dir, exist_ok=True) # ADDED directory creation check
            pred_df.to_csv(pred_output_path, index=False)
            print(f"DEBUG: Successfully completed pred_df.to_csv()") 
            
            # --- Check if file exists immediately after saving ---
            if os.path.exists(pred_output_path): # ADDED check
                print(f"DEBUG: os.path.exists confirms file exists at: {pred_output_path}") # ADDED check
            else: # ADDED check
                print(f"ERROR: os.path.exists CANNOT find file immediately after saving at: {pred_output_path}") # ADDED check
            # --- End check ---

            print(f"Detailed predictions saved to {pred_output_path}")
        except Exception as e: 
            print(f"ERROR: Failed to save CSV file: {e}") 
    else:
        print(f"Warning: Length mismatch between test_df ({len(test_df)}) and predictions ({len(labels)}). Skipping detailed prediction saving.")
    # --- End saving predictions ---

    # Update results dictionary with new metrics and inference time
    results = trainer.evaluate(tokenized_test) # Re-evaluate to get standard results dict structure
    results.update({
        'eval_accuracy_optimal': accuracy, # Use different keys for optimal threshold metrics
        'eval_f1_optimal': f1,
        'eval_f1_macro_optimal': f1_macro,
        'eval_f1_weighted_optimal': f1_weighted,
        'optimal_threshold': threshold,
        'inference_time_seconds': inference_time # Add inference time
    })
    
    return results

def plot_training_metrics(metrics_df):
    """Plot training metrics."""
    print("Available columns in metrics_df:", metrics_df.columns.tolist())
    
    plt.figure(figsize=(10, 5))
    
    # Check if we have the expected columns
    x_col = 'step' if 'step' in metrics_df.columns else 'epoch'
    
    # Plot accuracy if available
    plt.subplot(1, 2, 1)
    if 'accuracy' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['accuracy'], 'o-', label='Accuracy')
    elif 'eval_accuracy' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['eval_accuracy'], 'o-', label='Accuracy')
    plt.xlabel('Evaluation ' + x_col)
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Plot F1 score if available
    plt.subplot(1, 2, 2)
    if 'f1' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['f1'], 'o-', label='F1 Score')
    elif 'eval_f1' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['eval_f1'], 'o-', label='F1 Score')
    elif 'f1_weighted' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['f1_weighted'], 'o-', label='F1 Score')
    elif 'eval_f1_weighted' in metrics_df.columns:
        plt.plot(metrics_df[x_col], metrics_df['eval_f1_weighted'], 'o-', label='F1 Score')
    plt.xlabel('Evaluation ' + x_col)
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_metrics.png'))
    print(f"Training metrics plots saved to {os.path.join(LOG_DIR, 'training_metrics.png')}")

def extract_domain_features(text):
    """Extract healthcare domain-specific features from text."""
    features = {}
    
    # Convert list to string if needed
    if isinstance(text, list):
        text = ' '.join(text)
    
    # Check for side effect terminology
    features['contains_side_effect'] = 1 if any(term in text.lower() for term in 
        ['side effect', 'adverse', 'reaction', 'pain', 'symptom', 'hurt']) else 0
    
    # Check for benefit terminology
    features['contains_benefit'] = 1 if any(term in text.lower() for term in 
        ['help', 'improve', 'effective', 'relieve', 'better', 'good']) else 0
    
    # Check for severity indicators
    features['high_severity'] = 1 if any(term in text.lower() for term in 
        ['severe', 'extreme', 'terrible', 'worst', 'unbearable', 'awful']) else 0
    
    # Check for negation words near sentiment terms
    neg_pattern = r'(no|not|never|don\'t|doesn\'t|didn\'t|wasn\'t|weren\'t|haven\'t|hasn\'t|hadn\'t|can\'t|couldn\'t|won\'t|wouldn\'t|shouldn\'t)\s+(\w+\s+){0,3}(good|great|effective|helpful|better|improve)'
    features['contains_negation'] = 1 if re.search(neg_pattern, text.lower()) else 0
    
    # Check for comparative language
    features['contains_comparison'] = 1 if any(term in text.lower() for term in 
        ['better than', 'worse than', 'more than', 'less than', 'compared to']) else 0
        
    # Check for medication discontinuation
    features['discontinued'] = 1 if any(term in text.lower() for term in 
        ['stop', 'quit', 'discontinue', 'switched', 'change']) else 0
        
    return features

def enhance_dataset_with_features(df):
    """Add domain-specific features to dataset."""
    text_inputs = df["text_input"].tolist()
    
    # Extract domain features
    domain_features = [extract_domain_features(text) for text in text_inputs]
    domain_df = pd.DataFrame(domain_features)
    
    # Combine features with original dataframe
    for col in domain_df.columns:
        df[col] = domain_df[col].values
        
    return df

def augment_minority_class(df, n_samples=2):
    """Augment minority class samples using synonym replacement."""
    # Identify minority class samples
    positive_samples = df[df['label'] == 1]
    
    if len(positive_samples) == 0:
        return df
    
    print(f"Augmenting {len(positive_samples)} positive samples...")
    
    try:
        # Initialize augmenter
        aug = SynonymAug(aug_p=0.3)  # 30% of words will be replaced with synonyms
        
        augmented_samples = []
        for _, row in positive_samples.iterrows():
            original_text = row['text_input']
            
            # Create augmented versions
            for _ in range(n_samples):
                try:
                    augmented_text = aug.augment(original_text)
                    new_row = row.copy()
                    new_row['text_input'] = augmented_text
                    augmented_samples.append(new_row)
                except Exception as e:
                    print(f"Augmentation error: {e}")
                    continue
        
        # Combine original and augmented data
        augmented_df = pd.concat([df] + [pd.DataFrame([sample]) for sample in augmented_samples], ignore_index=True)
        print(f"Added {len(augmented_samples)} augmented positive samples")
        return augmented_df
    
    except Exception as e:
        print(f"Failed to perform text augmentation: {e}")
        return df

def find_optimal_threshold(trainer, eval_dataset):
    """Find optimal decision threshold for classification."""
    print("Finding optimal decision threshold...")
    
    # Get predictions
    predictions = trainer.predict(eval_dataset)
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1]
    labels = predictions.label_ids
    
    # Compute precision-recall curve and find the best F1 score
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    # Calculate F1 for each threshold
    f1_scores = []
    for precision, recall, threshold in zip(precisions[:-1], recalls[:-1], thresholds):
        if precision + recall > 0:  # Avoid division by zero
            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append((f1, threshold))
    
    # Sort by F1 score
    f1_scores.sort(reverse=True)
    
    if f1_scores:
        best_f1, best_threshold = f1_scores[0]
        print(f"Best threshold: {best_threshold:.4f}, F1: {best_f1:.4f}")
        
        # Plot precision-recall curve
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, 'b-', label='Precision-Recall curve')
        plt.axvline(x=recalls[np.where(thresholds >= best_threshold)[0][-1]], color='r', linestyle='--', 
                   label=f'Threshold={best_threshold:.2f}, F1={best_f1:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve with Optimal Threshold')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(LOG_DIR, 'precision_recall_curve.png'))
        
        return best_threshold
    else:
        print("Could not find optimal threshold.")
        return 0.5  # Default threshold

def apply_threshold(predictions, threshold=0.5):
    """Apply a custom threshold to prediction probabilities."""
    probs = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()[:, 1]
    return (probs >= threshold).astype(int)

def save_checkpoint_to_cpu(trainer, output_dir):
    """Save checkpoint by first moving model and optimizer states to CPU"""
    print("Saving checkpoint to CPU...")
    # Get model 
    model = trainer.model
    
    # Move model to CPU before saving
    model.to('cpu')
    
    # Save model only (skip optimizer)
    trainer.model = model
    trainer.save_model(output_dir)
    
    # Move model back to original device
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    trainer.model = model
    
    print(f"Model saved successfully to {output_dir}")

def main():
    # Load the CADEC ABSA datasets
    train_df, val_df, test_df = load_cadec_data()
    
    # --- Data Preparation and Balancing (Keep as is) ---
    # Find positive samples
    positive_samples = train_df[train_df['absa1'] == 2]
    non_positive_samples = train_df[train_df['absa1'] != 2]
    print(f"Original distribution: {len(positive_samples)} positive, {len(non_positive_samples)} non-positive")
    # 1. Text augmentation
    print("Performing text augmentation for the minority class...")
    train_df["text_input"] = train_df.apply(
        lambda row: f"{' '.join(row['tokens'])} [SEP] {row['absa2']} [SEP] {str(row['absa3'])}", axis=1
    )
    train_df["label"] = train_df["absa1"].apply(lambda x: 1 if x == 2 else 0)
    train_df = augment_minority_class(train_df, n_samples=3)
    # 2. SMOTE oversampling
    train_df = enhance_dataset_with_features(train_df)
    feature_cols = ["contains_side_effect", "contains_benefit", "high_severity", 
                   "contains_negation", "contains_comparison", "discontinued"]
    X_features = train_df[feature_cols].values
    y_labels = train_df["label"].values
    print("Applying SMOTE oversampling...")
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_features, y_labels)
        
        # Create balanced dataframe
        balanced_indices = []
        new_samples = []
        
        # Keep track of original samples
        original_indices = {}
        for i, label in enumerate(y_labels):
            if label not in original_indices:
                original_indices[label] = []
            original_indices[label].append(i)
            
        # Map resampled features back to original samples or create synthetic samples
        for i, (features, label) in enumerate(zip(X_resampled, y_resampled)):
            if i < len(train_df):
                balanced_indices.append(i)
            else:
                # This is a synthetic sample
                # Find closest original sample of the same class
                orig_indices = original_indices[label]
                orig_sample = train_df.iloc[orig_indices[0]].copy()  # Take the first one
                
                # Set the features for this synthetic sample
                for j, col in enumerate(feature_cols):
                    orig_sample[col] = features[j]
                
                new_samples.append(orig_sample)
        
        # Combine original and synthetic samples
        balanced_rows = list(train_df.iloc[balanced_indices].iterrows())
        for sample in new_samples:
            balanced_rows.append((None, sample))
        balanced_train_df = pd.DataFrame([row for _, row in balanced_rows])
        
        print(f"After SMOTE: {sum(y_resampled==1)} positive, {sum(y_resampled==0)} non-positive")
    except Exception as e:
        print(f"SMOTE error: {e}. Falling back to original oversampling.")
        oversampling_factor = max(1, len(non_positive_samples) // len(positive_samples)) # Ensure factor is at least 1
        oversampled_positives = pd.concat([positive_samples] * oversampling_factor) if len(positive_samples) > 0 else pd.DataFrame()
        balanced_train_df = pd.concat([non_positive_samples, oversampled_positives])
        print(f"Balanced distribution: {len(balanced_train_df[balanced_train_df['label']==1])} positive, {len(balanced_train_df[balanced_train_df['label']==0])} non-positive")
    # --- End Data Preparation ---

    # Prepare datasets for Hugging Face
    train_dataset = prepare_dataset_for_hf(balanced_train_df)
    val_dataset = prepare_dataset_for_hf(val_df)
    test_dataset = prepare_dataset_for_hf(test_df)
    
    # Initialize tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer)

    # --- Initialize Model, Training Args, Trainer ---
    # Training arguments (consistent for loading or training)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR, # Use OUTPUT_DIR for checkpoints etc.
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=WEIGHT_DECAY,
        learning_rate=LEARNING_RATE, # Add learning rate
        logging_dir=LOG_DIR, # Use LOG_DIR
        logging_steps=50,
        eval_strategy="steps", # Evaluate periodically
        eval_steps=100, # Evaluate every 100 steps
        save_strategy="steps", # Save based on steps
        save_steps=100, # Save checkpoint every 100 steps
        load_best_model_at_end=True, # Load the best model found during training
        metric_for_best_model="f1_weighted", # Use f1_weighted to select best model
        greater_is_better=True,
        save_total_limit=2 # Keep only the best and the latest checkpoint
    )

    # Calculate class weights
    labels = [train_dataset[i]["label"] for i in range(len(train_dataset))]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_weights_tensor = class_weights_tensor.to(device)
    print(f"Using device: {device}")
    print(f"Class weights: {class_weights}")

    # Initialize metrics callback
    metrics_callback = MetricsCallback(METRICS_FILE)

    # Check if a saved model exists
    if os.path.exists(SAVED_MODEL_PATH) and os.path.isdir(SAVED_MODEL_PATH):
        print(f"Loading pre-trained model from {SAVED_MODEL_PATH}")
        model = DistilBertForSequenceClassification.from_pretrained(SAVED_MODEL_PATH)
        metrics_df = None # No new training metrics when loading
        
        # Initialize Trainer with the loaded model
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train, # Still needed for trainer setup
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor,
            callbacks=[metrics_callback] # Add callback even when loading
        )

    else:
        print(f"No pre-trained model found at {SAVED_MODEL_PATH}. Training a new model...")
        # Model setup (only if training new)
        model = DistilBertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
        )
        
        # Initialize Trainer for training
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics,
            class_weights=class_weights_tensor,
            callbacks=[metrics_callback, EarlyStoppingCallback(early_stopping_patience=3)] # Add early stopping
        )

        # Train the model
        metrics_df = train_cadec_model(trainer, metrics_callback) # Pass callback

        # Save the best model after training
        print(f"Saving the best model to {SAVED_MODEL_PATH}")
        save_checkpoint_to_cpu(trainer, SAVED_MODEL_PATH) # Save the final best model

    # --- Evaluation ---
    # Plot training metrics if available (only if training occurred)
    if metrics_df is not None and not metrics_df.empty:
        plot_training_metrics(metrics_df)

    # Find optimal threshold on validation set
    optimal_threshold = find_optimal_threshold(trainer, tokenized_val)
    
    # Evaluate model on test set using optimal threshold
    # Ensure test_df has the required columns before passing
    test_df_eval = test_df.copy()
    if 'absa3' not in test_df_eval.columns:
         print("Warning: 'absa3' column missing in test_df. Adding dummy column.")
         # Add a dummy column if it's missing, or handle appropriately
         test_df_eval['absa3'] = 'Unknown' # Or load it correctly if possible
         
    test_results = evaluate_model(trainer, tokenized_test, test_df_eval, threshold=optimal_threshold) # Pass potentially modified test_df

    # Save final evaluation results
    results_path = os.path.join(OUTPUT_DIR, "enhanced_results.json")
    with open(results_path, 'w') as f:
        serializable_results = {}
        for k, v in test_results.items():
            if isinstance(v, (np.float32, np.float64)):
                serializable_results[k] = float(v)
            elif isinstance(v, (np.int32, np.int64)):
                 serializable_results[k] = int(v)
            else:
                 serializable_results[k] = v
        json.dump(serializable_results, f, indent=2)

    print("Processing complete!") # Updated message
    print(f"Final evaluation results saved to {results_path}")
    print(f"Checkpoints and logs are in {OUTPUT_DIR} and {LOG_DIR}")

# Add this block to call the main function when the script is run
if __name__ == "__main__":
    main()
