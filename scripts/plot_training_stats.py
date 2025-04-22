import json
import os
import matplotlib.pyplot as plt

# === Load metrics ===
metrics_path = "metrics.json"  # Change if it's stored elsewhere
if not os.path.exists(metrics_path):
    raise FileNotFoundError(f"Could not find metrics.json at: {metrics_path}")

with open(metrics_path, "r") as f:
    metrics = json.load(f)

# === Extract stats ===
epochs     = [m["epoch"] for m in metrics]
accuracies = [m["eval_accuracy"] for m in metrics]
f1_scores  = [m["eval_f1"] for m in metrics]
losses     = [m["eval_loss"] for m in metrics]

# === Create output folder ===
output_dir = "training_plots"
os.makedirs(output_dir, exist_ok=True)

# === Plot Accuracy ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, accuracies, marker='o')
plt.title("Evaluation Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_accuracy.png"))
plt.close()

# === Plot F1 Score ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, f1_scores, marker='o', color='green')
plt.title("Evaluation F1 Score per Epoch")
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_f1.png"))
plt.close()

# === Plot Loss ===
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o', color='red')
plt.title("Evaluation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eval_loss.png"))
plt.close()

print("✅ Training metric visualizations saved in the 'training_plots/' folder.")
