# Aspect-Based Sentiment Analysis (ABSA) for Healthcare Reviews

## Project Overview
This repository implements Aspect-Based Sentiment Analysis (ABSA) for healthcare reviews using the CADEC (CSIRO Adverse Drug Event Corpus) dataset. The project analyzes healthcare and medication reviews at a fine-grained level, extracting sentiments about specific aspects of patient experiences (effectiveness, side effects, dosage, etc.) rather than just overall sentiment.

## Features
- **Multiple Model Implementations**:
  - Domain-specific transformer-based ABSA models (DistilBERT)
  - Baseline sentiment models (Naive Bayes, SVM)
  - Performance comparison between approaches
- **Comprehensive Healthcare Dataset**:
  - CADEC dataset with medication reviews and adverse drug events
  - Aspect-level annotations and robust train/dev/test splits
  - Cleaned and preprocessed datasets
- **Evaluation & Visualization**:
  - Accuracy, F1, and other metrics
  - Performance and dataset visualizations
  - Detailed analysis of model results
- **Healthcare Domain Specificity**:
  - Medication efficacy and side effects analysis
  - Patient-reported outcomes interpretation
  - Healthcare-specific sentiment classification
- **Explainability**:
  - LIME and SHAP explanations for model predictions (see `lime_explanations/` and `shap_explanations/`)
- **Interactive Dashboard**:
  - Streamlit dashboard for interactive ABSA exploration (`streamlit_absa_dashboard.py`)
- **Data Augmentation & Feature Engineering**:
  - Scripts for augmenting minority classes and extracting domain-specific features

## Repository Structure
```
aspectRx/
├── Cadec_Data/            # Raw CADEC dataset and metadata
├── Dataset/               # Processed CADEC dataset with ABSA annotations
├── dataset_stats/         # Visualizations and statistics of dataset characteristics
├── evaluation/            # Evaluation scripts and metrics
├── lime_explanations/     # LIME HTML explanations for model predictions
├── logs/                  # Training logs
├── models/                # Model implementations
│   ├── absa/              # Transformer-based ABSA models
│   ├── baseline/          # Naive Bayes baseline models
│   └── svm/               # SVM baseline models
├── results/               # Model checkpoints and evaluation results
├── scripts/               # Utility scripts for visualization and analysis
├── shap_explanations/     # SHAP visualizations for model predictions
├── training_plots/        # Training performance visualizations
├── utils/                 # Utility functions for data processing
└── streamlit_absa_dashboard.py # Streamlit dashboard for ABSA
```

## Dataset
The project uses the CADEC (CSIRO Adverse Drug Event Corpus) dataset with healthcare and medication reviews containing aspect-based annotations:
- `tokens`: Tokenized medication review text
- `absa1`, `absa2`, `absa3`: Aspect annotations including:
  - Position indices of aspect terms
  - Aspect category (e.g., MEDICATION#EFFICACY, MEDICATION#SIDE-EFFECT, TREATMENT#DOSAGE)
  - Sentiment polarity (0=negative, 1=neutral, 2=positive)

The CADEC dataset is designed for research on adverse drug events and patient experiences with medications, making it ideal for healthcare-focused sentiment analysis applications.

## Model Architecture
The primary ABSA model uses DistilBERT, a lightweight transformer model, fine-tuned for aspect-based sentiment classification. The architecture includes:
- DistilBERT encoder for text representation
- Classification head for sentiment prediction
- Custom data preprocessing for aspect extraction
- Optional domain-specific feature engineering and data augmentation

## Data Processing
To process the CADEC dataset and generate ABSA-compatible files:
1. Download the CADEC dataset and place it in the `Cadec_Data/` folder
2. Run the processing script:
   ```bash
   python scripts/process_cadec_data.py
   ```

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/muhabdullahd/aspectRx.git
cd aspectRx

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data and spaCy model
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
```

## Usage

### Training the ABSA Model
```bash
cd models/absa
python absa-cadec.py
```

### Training Baseline Models
```bash
cd models/baseline
python train_baseline.py

# OR for SVM baseline
cd ../svm
python train_svm_baseline.py
```

### Evaluation
```bash
cd evaluation
python evaluate_metrics.py
```

### Visualizing Results
```bash
cd scripts
python plot_training_stats.py
```

### Running the Streamlit Dashboard
```bash
streamlit run streamlit_absa_dashboard.py
```

## Model Comparison with PyABSA

To evaluate the effectiveness of the domain-specific CADEC ABSA model, this project includes a script to compare its performance against a general-purpose ABSA model from the PyABSA library.

### Script: `scripts/compare_with_pyabsa.py`

This script performs the following steps:
1.  **Runs PyABSA**: Executes a pre-trained, general-purpose PyABSA model (Aspect Polarity Classification - APC) on the processed CADEC test set (`Dataset/cadec_absa_test.tsv`).
2.  **Loads Custom Model Results**: Reads the evaluation metrics (Accuracy and F1 score) of the custom-trained CADEC ABSA model from `results/cadec-absa/enhanced_results.json`.
3.  **Generates Comparison Plot**: Creates a bar chart comparing the Accuracy and F1 scores of the two models and saves it to `results/absa_comparison.png`.

### How to Run the Comparison

```bash
# Ensure you are in the aspectRx directory
# Activate your Python environment if you have one
# source absa_env/bin/activate 

python scripts/compare_with_pyabsa.py
```

This will output the metrics for both models and save the comparison plot.

## Results
- See `results/metrics.json` for evaluation metrics
- Visualizations in `training_plots/` and `dataset_stats/`
- LIME and SHAP explanations in their respective folders

## License
This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments
- Developed as a final project for a Machine Learning course
- Utilizes the CADEC (CSIRO Adverse Drug Event Corpus) dataset for healthcare and medication reviews

## Notes
- The CADEC dataset is not included in this repository due to size constraints. Please download it separately.
- For more details on dataset processing and utilities, see scripts and utils folders.