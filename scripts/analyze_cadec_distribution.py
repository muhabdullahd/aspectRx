#!/usr/bin/env python3
"""
Analyze CADEC dataset class distribution and visualize sentiment distributions.

This script loads the CADEC ABSA dataset and analyzes class distributions
to understand the extent of class imbalance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set plot style
plt.style.use('ggplot')
sns.set_palette("deep")

# Define file paths
DATASET_PATH = "../Dataset"
OUTPUT_PATH = "../dataset_stats/cadec_analysis"
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_datasets():
    """Load train, validation, and test datasets."""
    print("Loading CADEC datasets...")
    
    train_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_train.tsv", delimiter="\t")
    val_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_val.tsv", delimiter="\t")
    test_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_test.tsv", delimiter="\t")
    
    # Load full datasets for more detailed analysis
    train_full_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_train_full.tsv", delimiter="\t")
    val_full_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_val_full.tsv", delimiter="\t")
    test_full_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_test_full.tsv", delimiter="\t")
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df,
        'train_full': train_full_df,
        'val_full': val_full_df,
        'test_full': test_full_df
    }

def analyze_sentiment_distribution(datasets):
    """Analyze and visualize sentiment distribution in datasets."""
    print("\n=== Sentiment Distribution Analysis ===")
    
    # Create figure for sentiment distribution
    plt.figure(figsize=(15, 10))
    
    # Create subplots for each dataset
    for i, name in enumerate(['train', 'val', 'test']):
        df = datasets[name]
        sentiment_counts = df['absa1'].value_counts().sort_index()
        print(f"\n{name.capitalize()} set sentiment distribution:")
        
        # Calculate percentages
        total = len(df)
        for sentiment, count in sentiment_counts.items():
            sentiment_name = {0: "Negative", 1: "Neutral", 2: "Positive"}[sentiment]
            percentage = (count / total) * 100
            print(f"  {sentiment_name}: {count} samples ({percentage:.2f}%)")
        
        # Plot sentiment distribution
        plt.subplot(2, 2, i+1)
        ax = sentiment_counts.plot(kind='bar')
        plt.title(f"{name.capitalize()} Set Sentiment Distribution")
        plt.xlabel("Sentiment (0=Negative, 1=Neutral, 2=Positive)")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        
        # Add count labels on bars
        for j, p in enumerate(ax.patches):
            ax.annotate(str(p.get_height()), 
                      (p.get_x() + p.get_width()/2., p.get_height()), 
                      ha='center', va='bottom')
    
    # Plot combined sentiment distribution
    plt.subplot(2, 2, 4)
    combined_df = pd.concat([datasets['train'], datasets['val'], datasets['test']])
    combined_counts = combined_df['absa1'].value_counts().sort_index()
    ax = combined_counts.plot(kind='bar')
    plt.title("Combined Sentiment Distribution")
    plt.xlabel("Sentiment (0=Negative, 1=Neutral, 2=Positive)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    
    # Add count labels on bars
    for j, p in enumerate(ax.patches):
        ax.annotate(str(p.get_height()), 
                  (p.get_x() + p.get_width()/2., p.get_height()), 
                  ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/sentiment_distribution_all_sets.png")
    print(f"Saved sentiment distribution plot to {OUTPUT_PATH}/sentiment_distribution_all_sets.png")
    
    # Calculate binary sentiment distribution (for positive vs non-positive)
    plt.figure(figsize=(15, 5))
    
    for i, name in enumerate(['train', 'val', 'test']):
        df = datasets[name]
        # Convert to binary sentiment (0=non-positive, 1=positive)
        df['binary_sentiment'] = df['absa1'].apply(lambda x: 1 if x == 2 else 0)
        binary_counts = df['binary_sentiment'].value_counts().sort_index()
        
        # Print binary distribution
        print(f"\n{name.capitalize()} set binary sentiment distribution:")
        total = len(df)
        for sentiment, count in binary_counts.items():
            sentiment_name = {0: "Non-Positive", 1: "Positive"}[sentiment]
            percentage = (count / total) * 100
            print(f"  {sentiment_name}: {count} samples ({percentage:.2f}%)")
        
        # Plot binary sentiment distribution
        plt.subplot(1, 3, i+1)
        colors = ['#FF9999', '#99CC99']
        ax = binary_counts.plot(kind='bar', color=colors)
        plt.title(f"{name.capitalize()} Set Binary Sentiment")
        plt.xlabel("Sentiment (0=Non-Positive, 1=Positive)")
        plt.ylabel("Count")
        plt.xticks([0, 1], ['Non-Positive', 'Positive'])
        
        # Add count labels on bars
        for j, p in enumerate(ax.patches):
            ax.annotate(str(p.get_height()), 
                      (p.get_x() + p.get_width()/2., p.get_height()), 
                      ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/binary_sentiment_distribution.png")
    print(f"Saved binary sentiment distribution plot to {OUTPUT_PATH}/binary_sentiment_distribution.png")

def analyze_aspect_categories(datasets):
    """Analyze and visualize aspect categories in the datasets."""
    print("\n=== Aspect Category Analysis ===")
    
    # Get all aspect categories
    all_aspects = []
    for name in ['train_full', 'val_full', 'test_full']:
        df = datasets[name]
        aspects = df['aspect_category'].tolist()
        all_aspects.extend(aspects)
    
    # Count occurrences
    aspect_counts = Counter(all_aspects)
    
    # Print top aspect categories
    print("\nTop 10 aspect categories:")
    for aspect, count in aspect_counts.most_common(10):
        print(f"  {aspect}: {count} occurrences")
    
    # Plot aspect category distribution
    plt.figure(figsize=(12, 8))
    top_aspects = dict(aspect_counts.most_common(15))
    
    # Create horizontal bar plot
    plt.barh(list(top_aspects.keys()), list(top_aspects.values()))
    plt.title("Top 15 Aspect Categories in CADEC Dataset")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/top_aspect_categories.png")
    print(f"Saved top aspect categories plot to {OUTPUT_PATH}/top_aspect_categories.png")
    
    # Analyze sentiment by aspect category
    print("\n=== Sentiment by Aspect Category ===")
    
    # Combine all datasets
    combined_df = pd.concat([datasets['train_full'], datasets['val_full'], datasets['test_full']])
    
    # Get top 5 aspect categories
    top5_aspects = [aspect for aspect, _ in aspect_counts.most_common(5)]
    
    # Plot sentiment distribution for top 5 aspects
    plt.figure(figsize=(15, 10))
    
    for i, aspect in enumerate(top5_aspects):
        aspect_df = combined_df[combined_df['aspect_category'] == aspect]
        sentiment_counts = aspect_df['absa1'].value_counts().sort_index()
        
        # Print sentiment distribution for this aspect
        print(f"\nSentiment distribution for aspect '{aspect}':")
        total = len(aspect_df)
        for sentiment, count in sentiment_counts.items():
            sentiment_name = {0: "Negative", 1: "Neutral", 2: "Positive"}[sentiment]
            percentage = (count / total) * 100
            print(f"  {sentiment_name}: {count} samples ({percentage:.2f}%)")
        
        # Plot
        plt.subplot(2, 3, i+1)
        colors = ['#FF9999', '#66B2FF', '#99CC99']
        ax = sentiment_counts.plot(kind='bar', color=colors)
        plt.title(f"Sentiment for '{aspect}'")
        plt.xlabel("Sentiment (0=Negative, 1=Neutral, 2=Positive)")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        
        # Add count labels on bars
        for j, p in enumerate(ax.patches):
            ax.annotate(str(p.get_height()), 
                      (p.get_x() + p.get_width()/2., p.get_height()), 
                      ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/sentiment_by_top_aspects.png")
    print(f"Saved sentiment by top aspects plot to {OUTPUT_PATH}/sentiment_by_top_aspects.png")

def analyze_entity_types(datasets):
    """Analyze and visualize entity types in the datasets."""
    print("\n=== Entity Type Analysis ===")
    
    # Check if entity_type is present in the dataset
    if 'entity_type' not in datasets['train_full'].columns:
        print("Entity type information not available in the dataset")
        return
    
    # Combine all datasets
    combined_df = pd.concat([datasets['train_full'], datasets['val_full'], datasets['test_full']])
    
    # Count entity types
    entity_counts = combined_df['entity_type'].value_counts()
    
    # Print entity type distribution
    print("\nEntity type distribution:")
    for entity, count in entity_counts.items():
        percentage = (count / len(combined_df)) * 100
        print(f"  {entity}: {count} samples ({percentage:.2f}%)")
    
    # Plot entity type distribution
    plt.figure(figsize=(10, 6))
    ax = entity_counts.plot(kind='bar')
    plt.title("Entity Type Distribution in CADEC Dataset")
    plt.xlabel("Entity Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for i, p in enumerate(ax.patches):
        ax.annotate(str(p.get_height()), 
                  (p.get_x() + p.get_width()/2., p.get_height()), 
                  ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/entity_type_distribution.png")
    print(f"Saved entity type distribution plot to {OUTPUT_PATH}/entity_type_distribution.png")

def analyze_binary_classification_data():
    """Generate a balanced dataset for binary classification."""
    print("\n=== Binary Classification Dataset Analysis ===")
    
    # Load datasets
    train_df = pd.read_csv(f"{DATASET_PATH}/cadec_absa_train.tsv", delimiter="\t") 
    
    # Convert to binary sentiment (0=non-positive, 1=positive)
    train_df['binary_sentiment'] = train_df['absa1'].apply(lambda x: 1 if x == 2 else 0)
    
    # Count classes
    class_counts = train_df['binary_sentiment'].value_counts()
    print("\nBinary sentiment class counts in training data:")
    for cls, count in class_counts.items():
        cls_name = "Positive" if cls == 1 else "Non-Positive"
        print(f"  {cls_name} (class {cls}): {count} samples")
    
    # Calculate class imbalance ratio
    if 1 in class_counts and 0 in class_counts:
        imbalance_ratio = class_counts[0] / class_counts[1]
        print(f"\nClass imbalance ratio (Non-Positive:Positive): {imbalance_ratio:.2f}:1")
        
        # Calculate how many samples to add for balancing
        if imbalance_ratio > 1:
            samples_to_add = class_counts[0] - class_counts[1]
            print(f"To balance classes by oversampling: Need to add {samples_to_add} positive samples")
            print(f"To balance classes by undersampling: Need to remove {class_counts[0] - class_counts[1]} non-positive samples")
    
    # Generate sample code for balancing
    print("\nSample code to balance the dataset:")
    print("""
# Oversampling approach
positive_samples = train_df[train_df['binary_sentiment'] == 1]
non_positive_samples = train_df[train_df['binary_sentiment'] == 0]

# If you want to oversample the positive class
oversample_factor = len(non_positive_samples) // len(positive_samples)
oversampled_positive = positive_samples.sample(n=len(positive_samples) * oversample_factor, replace=True)
balanced_df = pd.concat([non_positive_samples, oversampled_positive])

# If you want to undersample the non-positive class
undersampled_non_positive = non_positive_samples.sample(n=len(positive_samples), replace=False)
balanced_df = pd.concat([undersampled_non_positive, positive_samples])
    """)

def main():
    print("=== CADEC Dataset Analysis ===\n")
    
    # Load datasets
    datasets = load_datasets()
    
    # Analyze sentiment distribution
    analyze_sentiment_distribution(datasets)
    
    # Analyze aspect categories
    analyze_aspect_categories(datasets)
    
    # Analyze entity types
    analyze_entity_types(datasets)
    
    # Analyze binary classification data and provide balancing suggestions
    analyze_binary_classification_data()
    
    print("\nAnalysis complete! Check the output directory for visualizations.")

if __name__ == "__main__":
    main()
