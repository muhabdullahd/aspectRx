#!/usr/bin/env python3
"""
CADEC Dataset Processor for ABSA

This script processes the CADEC dataset and converts it to ABSA format
compatible with existing models. It loads annotations from multiple sources,
combines them, and outputs training/validation/test sets.

Usage:
    python process_cadec_data.py
"""

import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils_cadec import (
    load_all_cadec_data, 
    split_cadec_dataset,
    save_cadec_absa_datasets,
    generate_dataset_statistics
)

def parse_args():
    parser = argparse.ArgumentParser(description='Process CADEC dataset for ABSA')
    parser.add_argument('--output-dir', default='Dataset', 
                      help='Directory to save processed datasets')
    parser.add_argument('--stats-dir', default='dataset_stats',
                      help='Directory to save dataset statistics')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                      help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                      help='Ratio of validation data')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                      help='Ratio of test data')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for dataset splitting')
    parser.add_argument('--generate-stats', action='store_true',
                      help='Generate dataset statistics and visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("cadec_processor")
    
    # 1. Load and process all CADEC data
    logger.info("Loading CADEC dataset...")
    cadec_df = load_all_cadec_data()
    
    if cadec_df.empty:
        logger.error("Failed to load CADEC dataset or no data found!")
        return 1
    
    # 2. Print basic dataset information
    logger.info(f"Loaded {len(cadec_df)} ABSA samples from CADEC dataset")
    logger.info(f"Number of unique reviews: {cadec_df['review_id'].nunique()}")
    
    # Show sentiment distribution
    sentiment_counts = cadec_df['absa1'].value_counts().sort_index()
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    for i, count in enumerate(sentiment_counts):
        logger.info(f"Sentiment '{sentiment_labels[i]}': {count} samples")
    
    # 3. Split the dataset
    logger.info("Splitting dataset into train/val/test sets...")
    train_df, val_df, test_df = split_cadec_dataset(
        cadec_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    # 4. Save the datasets
    logger.info(f"Saving processed datasets to {args.output_dir}...")
    save_cadec_absa_datasets(
        train_df, 
        val_df, 
        test_df, 
        output_folder=args.output_dir
    )
    
    # 5. Generate statistics if requested
    if args.generate_stats:
        logger.info("Generating dataset statistics and visualizations...")
        generate_dataset_statistics(cadec_df, output_folder=args.stats_dir)
    
    logger.info("Processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
