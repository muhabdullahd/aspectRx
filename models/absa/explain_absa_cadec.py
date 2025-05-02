"""
Explainability for CADEC ABSA Model using SHAP and LIME
"""
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np
import matplotlib.pyplot as plt

# Paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/cadec-absa/cadec-absa-model'))
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

# Load model and tokenizer
device = torch.device('cpu')
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# Prediction function for explainers
def predict_proba(texts):
    # SHAP may pass numpy arrays or other types; always convert to list of strings
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    if isinstance(texts, str):
        texts = [texts]
    if not isinstance(texts, list):
        texts = list(texts)
    # Ensure all elements are strings
    texts = [str(t) for t in texts]
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
    return probs

# Example texts (replace or extend as needed)
sample_texts = [
    "This medicine helped me a lot with my pain.",
    "I had severe side effects and had to stop taking it.",
    "No improvement after two weeks of use.",
    "Much better than my previous medication."
]

# ---- SHAP Explainability ----
print("\nRunning SHAP explainability...")
explainer = shap.Explainer(predict_proba, shap.maskers.Text(tokenizer))
shap_values = explainer(sample_texts)

# Create output directory for SHAP explanations
SHAP_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../shap_explanations'))
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# Save SHAP text explanations as PNG images
for i, val in enumerate(shap_values):
    plt.figure()
    shap.plots.text(val, display=False)
    png_path = os.path.join(SHAP_OUTPUT_DIR, f'shap_explanation_{i+1}.png')
    plt.savefig(png_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP explanation for sample {i+1} saved as {png_path}")

# Visualize SHAP for the first sample
shap.plots.text(shap_values[0])

# ---- LIME Explainability ----
print("\nRunning LIME explainability...")
class_names = ['Non-Positive', 'Positive']
lime_explainer = LimeTextExplainer(class_names=class_names)

# Create output directory for LIME explanations
LIME_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../lime_explanations'))
os.makedirs(LIME_OUTPUT_DIR, exist_ok=True)

for i, text in enumerate(sample_texts):
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        labels=[0, 1]
    )
    html_path = os.path.join(LIME_OUTPUT_DIR, f'lime_explanation_{i+1}.html')
    print(f"\nLIME explanation for sample {i+1} saved as {html_path}")
    exp.save_to_file(html_path)

print("\nExplainability complete. You can modify 'sample_texts' or adapt this script for batch explanations.")
