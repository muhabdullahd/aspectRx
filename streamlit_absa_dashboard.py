import streamlit as st
import sys
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import numpy as np
import spacy
from lime.lime_text import LimeTextExplainer
import os

# Prevent Streamlit from watching torch.classes
if hasattr(torch, '_classes'):
    sys.modules['torch._classes'] = None

# Load spaCy model for aspect extraction
nlp = spacy.load("en_core_web_sm")

# Model paths
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results/cadec-absa/cadec-absa-model'))
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256

# Load model and tokenizer
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    return model, tokenizer, device

model, tokenizer, device = load_model()

# Predict sentiment for a given text-aspect pair
def predict_sentiment(text, aspect):
    input_text = f"{text} [SEP] {aspect} [SEP] "
    inputs = tokenizer(input_text, padding='max_length', truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    return pred, probs

# LIME explanation
def lime_explanation(text, aspect):
    input_text = f"{text} [SEP] {aspect} [SEP] "
    class_names = ['Non-Positive', 'Positive']
    explainer = LimeTextExplainer(class_names=class_names)
    def predict_proba(texts):
        inputs = tokenizer(list(texts), padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        return probs
    exp = explainer.explain_instance(input_text, predict_proba, num_features=8, labels=[0, 1])
    return exp.as_html(), exp.as_list(label=exp.available_labels()[0])

# Aspect extraction (simple noun chunks)
def extract_aspects(text):
    doc = nlp(text)
    aspects = list(set([chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) > 2]))
    if not aspects:
        aspects = [sent.text.strip() for sent in doc.sents]
    return aspects

# Streamlit UI
st.title("Healthcare ABSA Demo: Aspect-wise Sentiment & Explanation")
st.write("Enter a healthcare review. The app will extract aspects, predict sentiment, and provide explanations.")

user_review = st.text_area("Enter a medication/healthcare review:", height=120)

if st.button("Analyze") and user_review.strip():
    st.subheader("Aspect-wise Sentiment Analysis")
    aspects = extract_aspects(user_review)
    if not aspects:
        st.warning("No aspects found in the review.")
    for aspect in aspects:
        pred, probs = predict_sentiment(user_review, aspect)
        sentiment = "Positive" if pred == 1 else "Non-Positive"
        st.markdown(f"**Aspect:** `{aspect}`")
        st.markdown(f"**Predicted Sentiment:** {sentiment} (Prob: {probs[pred]:.2f})")
        with st.expander("Show Explanation"):
            html, weights = lime_explanation(user_review, aspect)
            st.components.v1.html(html, height=350)
else:
    st.info("Enter a review and click Analyze to see results.")
