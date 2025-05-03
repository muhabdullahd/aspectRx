"""
Generate domain-specific performance analysis and report runtime benchmarks.

Reads detailed predictions and final metrics to:
1. Calculate Accuracy and F1 (macro) per aspect category.
2. Plot Accuracy and F1 per aspect category.
3. Print a summary table of metrics per aspect category.
4. Report the inference runtime benchmark.
5. Save plots to results/domain_analysis/
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Configuration ===
PREDICTIONS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/cadec-absa/test_predictions_with_aspects.csv'))
METRICS_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/cadec-absa/enhanced_results.json'))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/domain_analysis/'))

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Load Data ===
try:
    pred_df = pd.read_csv(PREDICTIONS_FILE)
except FileNotFoundError:
    print(f"Error: Predictions file not found at {PREDICTIONS_FILE}")
    print("Please ensure the 'absa-cadec.py' script has been run successfully after the latest changes.")
    exit()

try:
    with open(METRICS_FILE, 'r') as f:
        metrics_data = json.load(f)
except FileNotFoundError:
    print(f"Error: Metrics file not found at {METRICS_FILE}")
    print("Please ensure the 'absa-cadec.py' script has been run successfully.")
    exit()

# === Analyze Performance per Aspect Category ===
print("\n=== Performance Analysis by Aspect Category ===")

aspect_groups = pred_df.groupby('aspect_category')
results = []

for name, group in aspect_groups:
    true_labels = group['true_label']
    pred_labels = group['predicted_label']
    
    if len(true_labels) == 0:
        continue

    accuracy = accuracy_score(true_labels, pred_labels)
    # Use macro F1 to handle potential label imbalance within an aspect
    f1_macro = f1_score(true_labels, pred_labels, average='macro', zero_division=0) 
    support = len(true_labels)

    results.append({
        'Aspect Category': name,
        'Accuracy': accuracy,
        'F1 (Macro)': f1_macro,
        'Support': support
    })

results_df = pd.DataFrame(results).sort_values(by='Support', ascending=False)

# --- Filter for Top N aspects for plotting ---
N_TOP_ASPECTS = 15
plot_df = results_df.head(N_TOP_ASPECTS)

# Print summary table (show all aspects)
print("\n--- Metrics per Aspect Category (All) ---")
print(results_df.to_string(index=False, float_format="%.4f"))

# --- Generate Plots (Top N Aspects) ---
print(f"\n--- Generating plots for Top {N_TOP_ASPECTS} Aspects ---")

# Plot Accuracy per Aspect
plt.figure(figsize=(10, 8)) # Adjusted figure size
# Use plot_df for plotting
sns.barplot(data=plot_df, x='Accuracy', y='Aspect Category', hue='Aspect Category', palette='viridis', legend=False)
plt.title(f'Top {N_TOP_ASPECTS} Aspects: Accuracy per Category (Domain)')
plt.xlabel('Accuracy')
plt.ylabel('Aspect Category')
plt.xlim(0, 1)
plt.tight_layout()
acc_plot_path = os.path.join(OUTPUT_DIR, 'accuracy_by_aspect.png')
plt.savefig(acc_plot_path)
print(f"\nAccuracy plot saved to: {acc_plot_path}")
plt.close()

# Plot F1 (Macro) per Aspect
plt.figure(figsize=(10, 8)) # Adjusted figure size
# Use plot_df for plotting
sns.barplot(data=plot_df, x='F1 (Macro)', y='Aspect Category', hue='Aspect Category', palette='magma', legend=False)
plt.title(f'Top {N_TOP_ASPECTS} Aspects: F1 Score (Macro) per Category (Domain)')
plt.xlabel('F1 Score (Macro)')
plt.ylabel('Aspect Category')
plt.xlim(0, 1)
plt.tight_layout()
f1_plot_path = os.path.join(OUTPUT_DIR, 'f1_by_aspect.png')
plt.savefig(f1_plot_path)
print(f"F1 plot saved to: {f1_plot_path}")
plt.close()

# === Report Runtime Benchmark ===
print("\n=== Runtime Benchmark ===")
inference_time = metrics_data.get('inference_time_seconds')
optimal_threshold = metrics_data.get('optimal_threshold')
num_test_samples = len(pred_df)

if inference_time is not None:
    print(f"Model: Custom CADEC ABSA (DistilBERT-based)")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print(f"Test Set Size: {num_test_samples} samples")
    print(f"Total Inference Time: {inference_time:.2f} seconds")
    if num_test_samples > 0:
        time_per_sample = (inference_time / num_test_samples) * 1000 # in milliseconds
        print(f"Average Time per Sample: {time_per_sample:.2f} ms")
else:
    print("Inference time not found in metrics file.")
    # Add debug print for available keys
    print(f"DEBUG: Keys available in metrics_data: {list(metrics_data.keys())}")

print("\nDomain analysis complete.")

