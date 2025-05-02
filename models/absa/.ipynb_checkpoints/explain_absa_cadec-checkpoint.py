"""
Explainability for CADEC ABSA Model using SHAP and LIME
"""
import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import shap
from lime.lime_text import LimeTextExplainer
import numpy as np

# Paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results/cadec-absa/cadec-absa-model'))
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

# Load model and tokenizer
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

# Visualize SHAP for the first sample
shap.plots.text(shap_values[0])

# ---- LIME Explainability ----
print("\nRunning LIME explainability...")
class_names = ['Non-Positive', 'Positive']
lime_explainer = LimeTextExplainer(class_names=class_names)

for i, text in enumerate(sample_texts):
    exp = lime_explainer.explain_instance(
        text,
        predict_proba,
        num_features=10,
        labels=[0, 1]
    )
    print(f"\nLIME explanation for sample {i+1}:")
    exp.show_in_notebook(text=True)
    # To save as HTML: exp.save_to_file(f'lime_explanation_{i+1}.html')

print("\nExplainability complete. You can modify 'sample_texts' or adapt this script for batch explanations.")
