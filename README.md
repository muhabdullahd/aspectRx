# Aspect-Based Sentiment Analysis (ABSA) for Healthcare Reviews

## Project Overview
This repository contains an implementation of Aspect-Based Sentiment Analysis (ABSA) for healthcare reviews using the CADEC (CSIRO Adverse Drug Event Corpus) dataset. The project aims to analyze healthcare and medication reviews at a fine-grained level, extracting sentiments about specific aspects of patient experiences with medications (effectiveness, side effects, dosage, etc.) rather than just the overall sentiment of the review.

## Features
- **Multiple Model Implementations**:
  - Domain-specific transformer-based ABSA models using DistilBERT
  - Baseline sentiment models (Naive Bayes, SVM)
  - Performance comparison between approaches
- **Comprehensive Healthcare Dataset**:
  - CADEC dataset with medication reviews and adverse drug events
  - Healthcare-specific aspect-level annotations
  - Train/Dev/Test splits for robust evaluation
  - Cleaned and preprocessed versions of datasets
- **Evaluation Framework**:
  - Accuracy, F1, and other relevant metrics
  - Performance visualization tools
  - Detailed analysis of model results
- **Healthcare Domain Specificity**:
  - Medication efficacy and side effects analysis
  - Patient-reported outcomes interpretation
  - Healthcare-specific sentiment classification

## Repository Structure
```
dsa-absa/
├── Dataset/               # Restaurant reviews dataset with ABSA annotations
├── dataset_stats/         # Visualizations of dataset characteristics
├── evaluation/            # Evaluation scripts and metrics
├── logs/                  # Training logs
├── models/                # Model implementations
│   ├── absa/              # Transformer-based ABSA models
│   ├── baseline/          # Naive Bayes baseline models
│   └── svm/               # SVM baseline models
├── results/               # Model checkpoints and evaluation results
├── saved_model/           # Best performing models
├── scripts/               # Utility scripts for visualization and analysis
├── training_plots/        # Training performance visualizations
└── utils/                 # Utility functions for data processing
```

## Dataset
The project uses the CADEC (CSIRO Adverse Drug Event Corpus) dataset with healthcare and medication reviews containing aspect-based annotations in the format:
- `tokens`: Tokenized medication review text
- `absa1`, `absa2`, `absa3`: Aspect annotations including:
  - Position indices of aspect terms
  - Aspect category (e.g., MEDICATION#EFFICACY, MEDICATION#SIDE-EFFECT, TREATMENT#DOSAGE)
  - Sentiment polarity (0=negative, 1=neutral, 2=positive)

The CADEC dataset is specifically designed for research on adverse drug events and patient experiences with medications, making it ideal for healthcare-focused sentiment analysis applications.

## Model Architecture
The primary ABSA model uses DistilBERT, a lightweight transformer model, fine-tuned for aspect-based sentiment classification. The architecture includes:
- DistilBERT encoder for text representation
- Classification head for sentiment prediction
- Custom data preprocessing for aspect extraction


### Data Processing

To process the CADEC dataset and generate ABSA-compatible files:

1. Download the CADEC dataset and place it in the `Cadec_Data/` folder
2. Run the processing script:

## Setup and Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/dsa-absa.git
cd dsa-absa

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
python absa-1.py
```

### Training Baseline Models
```bash
cd models/baseline
python train_baseline.py

# OR for SVM baseline
cd models/svm
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

## Results
The transformer-based ABSA model achieves superior performance compared to baseline models:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| DistilBERT ABSA | ~0.85 | ~0.84 |
| Naive Bayes | ~0.70 | ~0.68 |
| SVM | ~0.75 | ~0.74 |

## License
This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments
- This project was developed as a final project for the Machine Learning course.
- The dataset is based on the SemEval-2016 restaurant reviews dataset.

### Folder Structure

- `utils/data_utils_cadec.py`: Utilities to process CADEC annotations
- `scripts/process_cadec_data.py`: Script to generate processed datasets
- `dataset_stats/`: Visualizations and statistics of the processed dataset

### Output Files

The script generates these files in the `Dataset/` folder:
- `cadec_absa_train.tsv`: Training set
- `cadec_absa_val.tsv`: Validation set
- `cadec_absa_test.tsv`: Test set
- Additional `*_full.tsv` files with extended metadata

### Note

The CADEC dataset is not included in this repository due to size constraints. Please download it separately.