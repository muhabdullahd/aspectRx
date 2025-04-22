"""
CADEC Dataset Processing Utilities for ABSA
This module contains utilities to process the CADEC (CSIRO Adverse Drug Event Corpus)
dataset and convert it into a format suitable for Aspect-Based Sentiment Analysis (ABSA).

The CADEC dataset contains patient medication reviews with annotations for adverse drug
events. This utility combines information from multiple annotation sources:
1. Original patient reviews (cadec/text)
2. Entity and type information (CADEC.v1/Original)
3. MedDRA mappings for standardized medical terminology (cadec/meddra)
4. SNOMED CT mappings (CADEC.v1/AMT-SCT)
"""

import os
import re
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define paths
CADEC_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Cadec_Data')
CADEC_TEXT_PATH = os.path.join(CADEC_DATA_PATH, 'data', 'cadec', 'text')
CADEC_ORIGINAL_PATH = os.path.join(CADEC_DATA_PATH, 'data', 'CADEC.v1', 'Original')
CADEC_MEDDRA_PATH = os.path.join(CADEC_DATA_PATH, 'data', 'cadec', 'meddra')
CADEC_SCT_PATH = os.path.join(CADEC_DATA_PATH, 'data', 'CADEC.v1', 'AMT-SCT')

# Define aspect categories based on entity types
ENTITY_TO_ASPECT = {
    'ADR': 'MEDICATION#SIDE-EFFECT',
    'Drug': 'MEDICATION#GENERAL',
    'Disease': 'CONDITION#DISEASE',
    'Symptom': 'CONDITION#SYMPTOM',
    'Finding': 'CONDITION#FINDING'
}

# Define default sentiment mappings (will be adjusted by context)
ENTITY_TO_SENTIMENT = {
    'ADR': 0,  # Negative by default
    'Drug': 1,  # Neutral by default
    'Disease': 1,  # Neutral by default
    'Symptom': 0,  # Negative by default
    'Finding': 1   # Neutral by default
}

def get_review_ids():
    """
    Get all unique review IDs from the text folder.
    
    Returns:
        list: Review IDs (file names without extension)
    """
    text_files = glob.glob(os.path.join(CADEC_TEXT_PATH, '*.txt'))
    review_ids = [os.path.basename(f).replace('.txt', '') for f in text_files]
    return sorted(review_ids)

def load_review_text(review_id):
    """
    Load the original patient review text.
    
    Args:
        review_id (str): The review ID (e.g., 'ARTHROTEC.1')
        
    Returns:
        str: The review text or None if file not found
    """
    text_path = os.path.join(CADEC_TEXT_PATH, f'{review_id}.txt')
    if not os.path.exists(text_path):
        logger.warning(f"Review text file not found: {text_path}")
        return None
        
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error reading review text {text_path}: {str(e)}")
        return None

def load_entity_annotations(review_id):
    """
    Load entity annotations with types from CADEC.v1/Original.
    
    Args:
        review_id (str): The review ID (e.g., 'ARTHROTEC.1')
        
    Returns:
        list: A list of dictionaries with entity annotations
    """
    ann_path = os.path.join(CADEC_ORIGINAL_PATH, f'{review_id}.ann')
    annotations = []
    
    if not os.path.exists(ann_path):
        logger.warning(f"Original annotation file not found: {ann_path}")
        return annotations
        
    try:
        with open(ann_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if line.startswith('T'):  # Entity annotation
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    entity_id = parts[0]
                    entity_info = parts[1].split(' ')
                    entity_type = entity_info[0]  # ADR, Drug, Disease, Symptom
                    
                    # Handle cases where there might be spaces in entity type
                    if len(entity_info) < 3:
                        continue
                    
                    # Handle discontinuous spans (positions with semicolons)
                    position_info = ' '.join(entity_info[1:])
                    
                    # Check if this is a discontinuous span (contains semicolon)
                    if ';' in position_info:
                        # For discontinuous spans, we'll use the first start and last end position
                        # This approach treats the discontinuous annotation as a single span including
                        # the text between discontinuous parts
                        spans = position_info.split(';')
                        try:
                            # Get first start position
                            start_pos = int(spans[0].split()[0])
                            # Get last end position
                            end_pos = int(spans[-1].split()[-1])
                        except (ValueError, IndexError):
                            logger.warning(f"Complex discontinuous span in {ann_path}, line: {line}")
                            continue
                    else:
                        # Regular continuous span
                        try:
                            start_pos = int(entity_info[1])
                            end_pos = int(entity_info[2])
                        except ValueError:
                            logger.warning(f"Invalid position values in {ann_path}, line: {line}")
                            continue
                        
                    entity_text = parts[2]
                    
                    # Look for annotator notes (normalized terms)
                    normalized_term = None
                    for j in range(i+1, len(lines)):
                        if lines[j].startswith(f'#{entity_id.replace("T", "")}'):
                            note_parts = lines[j].strip().split('\t')
                            if len(note_parts) >= 3:
                                normalized_term = note_parts[2]
                            break
                        elif lines[j].startswith('T'):  # Next entity
                            break
                    
                    annotations.append({
                        'id': entity_id,
                        'type': entity_type,
                        'text': entity_text,
                        'start': start_pos,
                        'end': end_pos,
                        'normalized': normalized_term
                    })
        
        return annotations
    except Exception as e:
        logger.error(f"Error reading annotation file {ann_path}: {str(e)}")
        return []

def load_meddra_mappings(review_id):
    """
    Load MedDRA concept mappings.
    
    Args:
        review_id (str): The review ID
        
    Returns:
        dict: A dictionary mapping entity positions to MedDRA codes
    """
    meddra_path = os.path.join(CADEC_MEDDRA_PATH, f'{review_id}.ann')
    mappings = {}
    
    if not os.path.exists(meddra_path):
        return mappings
        
    try:
        with open(meddra_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    concept_info = parts[1].split(' ')
                    if len(concept_info) >= 3:
                        try:
                            meddra_code = concept_info[0]
                            start_pos = int(concept_info[1])
                            end_pos = int(concept_info[2])
                            key = f"{start_pos}_{end_pos}"
                            mappings[key] = meddra_code
                        except (ValueError, IndexError):
                            continue
        
        return mappings
    except Exception as e:
        logger.error(f"Error reading MedDRA file {meddra_path}: {str(e)}")
        return {}

def adjust_sentiment_by_context(entity_text, entity_type, context):
    """
    Adjust sentiment based on contextual clues in the review.
    
    Args:
        entity_text (str): The text of the entity mention
        entity_type (str): The type of entity (ADR, Drug, etc.)
        context (str): The surrounding context
        
    Returns:
        int: Adjusted sentiment (0=negative, 1=neutral, 2=positive) or None if no adjustment
    """
    # Don't adjust if entity text is empty
    if not entity_text or not context:
        return None
        
    # Convert to lowercase for case-insensitive matching
    entity_lower = entity_text.lower()
    context_lower = context.lower()
    
    # 1. Check for explicit negation patterns
    negation_patterns = [
        r'\bno\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
        r'\bnot\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
        r"don't\s+(?:\w+\s+){0,3}" + re.escape(entity_lower),
        r"doesn't\s+(?:\w+\s+){0,3}" + re.escape(entity_lower),
        r"didn't\s+(?:\w+\s+){0,3}" + re.escape(entity_lower),
        r'\bwithout\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
        r'\bfree\s+of\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
        r'\bno\s+more\s+(?:\w+\s+){0,3}' + re.escape(entity_lower)
    ]
    
    # Check for specific phrases that indicate absence of an ADR (positive sentiment)
    if entity_type == 'ADR' or entity_type == 'Symptom':
        for pattern in negation_patterns:
            if re.search(pattern, context_lower):
                return 2  # Flip negative to positive (absence of ADR is good)
    
    # 2. Check for intensity modifiers
    if entity_type == 'ADR' or entity_type == 'Symptom':
        intensity_patterns = [
            r'severe\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
            r'extreme\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
            r'terrible\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
            r'worst\s+(?:\w+\s+){0,3}' + re.escape(entity_lower),
            r'unbearable\s+(?:\w+\s+){0,3}' + re.escape(entity_lower)
        ]
        
        for pattern in intensity_patterns:
            if re.search(pattern, context_lower):
                return 0  # Emphasize negative sentiment for ADRs
    
    # 3. Check for improvement or relief patterns
    improvement_patterns = [
        r'(?:help|helps|helped|helping)(?:\s+\w+){0,5}\s+' + re.escape(entity_lower),
        r'(?:improve|improves|improved|improving)(?:\s+\w+){0,5}\s+' + re.escape(entity_lower),
        r'(?:relieve|relieves|relieved|relieving)(?:\s+\w+){0,5}\s+' + re.escape(entity_lower),
        r'(?:reduce|reduces|reduced|reducing)(?:\s+\w+){0,5}\s+' + re.escape(entity_lower),
        r'better\s+(?:\w+\s+){0,5}' + re.escape(entity_lower),
        r'gone\s+(?:\w+\s+){0,3}' + re.escape(entity_lower)
    ]
    
    for pattern in improvement_patterns:
        if re.search(pattern, context_lower):
            if entity_type == 'Disease' or entity_type == 'Symptom':
                return 2  # Positive (symptoms improving is good)
    
    # 4. Check for efficacy mentions for drugs
    if entity_type == 'Drug':
        efficacy_positive = [
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:work|works|worked|working)',
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:help|helps|helped|helping)',
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:effective|good|great|excellent)',
            r'happy\s+(?:\w+\s+){0,5}(?:with|about)\s+(?:\w+\s+){0,3}' + re.escape(entity_lower)
        ]
        
        efficacy_negative = [
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:not\s+work|doesn\'t\s+work|didn\'t\s+work)',
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:stop|stopped|stopping)',
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:useless|ineffective|bad)',
            r'(?:' + re.escape(entity_lower) + r')\s+(?:\w+\s+){0,5}(?:not\s+help|doesn\'t\s+help|didn\'t\s+help)'
        ]
        
        for pattern in efficacy_positive:
            if re.search(pattern, context_lower):
                return 2  # Positive sentiment for effective drugs
                
        for pattern in efficacy_negative:
            if re.search(pattern, context_lower):
                return 0  # Negative sentiment for ineffective drugs
    
    # No adjustment needed
    return None

def simple_tokenize(text):
    """
    Simple tokenization function that preserves medication names and handles common
    punctuation in patient reviews.
    
    Args:
        text (str): Text to tokenize
        
    Returns:
        list: List of tokens
    """
    if not text:
        return []
        
    # Replace punctuation with space + punctuation to ensure separation
    for punct in '.,:;!?()[]{}"\'':
        text = text.replace(punct, f' {punct} ')
    
    # Split by whitespace and filter empty strings
    tokens = [token for token in text.split() if token.strip()]
    return tokens

def process_review_for_absa(review_id):
    """
    Process a single review for aspect-based sentiment analysis.
    
    Args:
        review_id (str): The review ID
        
    Returns:
        list: List of ABSA data samples for this review
    """
    # 1. Get review text
    review_text = load_review_text(review_id)
    if not review_text:
        return []
    
    # 2. Get entity annotations
    entities = load_entity_annotations(review_id)
    if not entities:
        return []
    
    # 3. Get MedDRA mappings
    meddra_mappings = load_meddra_mappings(review_id)
    
    # 4. Process entities into ABSA format
    absa_samples = []
    
    for entity in entities:
        # Map entity type to aspect category
        aspect_category = ENTITY_TO_ASPECT.get(entity['type'], 'OTHER')
        
        # Get default sentiment based on entity type
        sentiment = ENTITY_TO_SENTIMENT.get(entity['type'], 1)  # Default to neutral
        
        # Get context around entity (100 chars before and after)
        context_window = 100
        start_idx = max(0, entity['start'] - context_window)
        end_idx = min(len(review_text), entity['end'] + context_window)
        context = review_text[start_idx:end_idx]
        
        # Adjust sentiment based on context
        adjusted_sentiment = adjust_sentiment_by_context(entity['text'], entity['type'], context)
        if adjusted_sentiment is not None:
            sentiment = adjusted_sentiment
        
        # Get MedDRA code if available
        meddra_code = meddra_mappings.get(f"{entity['start']}_{entity['end']}", None)
        
        # Create normalized term (from annotator notes or original entity text)
        normalized_term = entity['normalized'] if entity['normalized'] else entity['text']
        
        # Build aspect term to include entity type and normalized term
        aspect_term = f"{entity['type']}:{normalized_term}"
        
        # Add the normalized term to the aspect category if available
        if entity['normalized']:
            subcategory = re.sub(r'[^A-Z0-9]', '', entity['normalized'].upper())
            aspect_category = f"{aspect_category}:{subcategory}"
        
        # Build ABSA sample
        absa_samples.append({
            'review_id': review_id,
            'text': review_text,
            'tokens': simple_tokenize(review_text),
            'aspect_text': entity['text'],  # Original mention text
            'aspect_term': aspect_term,  # Enhanced aspect term with entity type
            'aspect_category': aspect_category,  # Category based on entity type
            'absa1': sentiment,  # 0=negative, 1=neutral, 2=positive
            'absa2': aspect_category,  # Category (matches aspect_category for consistency)
            'absa3': aspect_term,  # Term (matches aspect_term for consistency)
            'entity_type': entity['type'],  # Original entity type
            'entity_start': entity['start'],  # Character offset start
            'entity_end': entity['end'],  # Character offset end
            'normalized_term': normalized_term,  # Normalized term
            'meddra_code': meddra_code  # MedDRA code if available
        })
    
    return absa_samples

def load_all_cadec_data():
    """
    Load and process all CADEC data for ABSA.
    
    Returns:
        pandas.DataFrame: DataFrame with all ABSA samples
    """
    logger.info("Loading CADEC data for ABSA...")
    review_ids = get_review_ids()
    logger.info(f"Found {len(review_ids)} reviews to process")
    
    all_samples = []
    for i, review_id in enumerate(review_ids):
        if i % 100 == 0:
            logger.info(f"Processing review {i+1}/{len(review_ids)}: {review_id}")
        samples = process_review_for_absa(review_id)
        all_samples.extend(samples)
    
    logger.info(f"Processed {len(all_samples)} ABSA samples from {len(review_ids)} reviews")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_samples)
    return df

def split_cadec_dataset(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Split CADEC dataset into train/val/test sets.
    Split by review_id to ensure all aspects from the same review are in the same split.
    
    Args:
        df (pandas.DataFrame): DataFrame with ABSA samples
        train_ratio (float): Ratio for training set
        val_ratio (float): Ratio for validation set
        test_ratio (float): Ratio for test set
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10
    
    # Get unique review IDs
    review_ids = df['review_id'].unique()
    
    # Split review IDs
    train_val_ids, test_ids = train_test_split(
        review_ids, 
        test_size=test_ratio, 
        random_state=random_state
    )
    
    # Calculate proportion for train from train_val
    train_prop = train_ratio / (train_ratio + val_ratio)
    
    train_ids, val_ids = train_test_split(
        train_val_ids,
        train_size=train_prop,
        random_state=random_state
    )
    
    # Create DataFrames based on split IDs
    train_df = df[df['review_id'].isin(train_ids)].copy()
    val_df = df[df['review_id'].isin(val_ids)].copy()
    test_df = df[df['review_id'].isin(test_ids)].copy()
    
    logger.info(f"Train set: {len(train_df)} samples from {len(train_ids)} reviews")
    logger.info(f"Validation set: {len(val_df)} samples from {len(val_ids)} reviews")
    logger.info(f"Test set: {len(test_df)} samples from {len(test_ids)} reviews")
    
    return train_df, val_df, test_df

def save_cadec_absa_datasets(train_df, val_df, test_df, output_folder="Dataset"):
    """
    Save processed CADEC datasets in TSV format.
    
    Args:
        train_df (pandas.DataFrame): Training data
        val_df (pandas.DataFrame): Validation data
        test_df (pandas.DataFrame): Test data
        output_folder (str): Folder to save datasets
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Define columns to save (matching the format of rest16_quad datasets)
    cols_to_save = ['text', 'tokens', 'absa1', 'absa2', 'absa3']
    
    # Save datasets
    train_df[cols_to_save].to_csv(os.path.join(output_folder, "cadec_absa_train.tsv"), sep="\t", index=False)
    val_df[cols_to_save].to_csv(os.path.join(output_folder, "cadec_absa_val.tsv"), sep="\t", index=False)
    test_df[cols_to_save].to_csv(os.path.join(output_folder, "cadec_absa_test.tsv"), sep="\t", index=False)
    
    # Save full datasets with all metadata (for further analysis)
    train_df.to_csv(os.path.join(output_folder, "cadec_absa_train_full.tsv"), sep="\t", index=False)
    val_df.to_csv(os.path.join(output_folder, "cadec_absa_val_full.tsv"), sep="\t", index=False)
    test_df.to_csv(os.path.join(output_folder, "cadec_absa_test_full.tsv"), sep="\t", index=False)
    
    logger.info(f"CADEC ABSA datasets saved to {output_folder}")
    logger.info(f"Primary files: cadec_absa_train.tsv, cadec_absa_val.tsv, cadec_absa_test.tsv")
    logger.info(f"Full metadata files: cadec_absa_train_full.tsv, cadec_absa_val_full.tsv, cadec_absa_test_full.tsv")

def generate_dataset_statistics(df, output_folder="dataset_stats"):
    """
    Generate and save statistics about the CADEC ABSA dataset.
    
    Args:
        df (pandas.DataFrame): The ABSA data
        output_folder (str): Folder to save statistics
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Sentiment distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['absa1'].value_counts().sort_index()
    labels = ['Negative', 'Neutral', 'Positive']
    colors = ['#FF9999', '#66B2FF', '#99CC99']
    
    ax = sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette=colors)
    for i, v in enumerate(sentiment_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    
    plt.title('Sentiment Distribution in CADEC ABSA Dataset')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks([0, 1, 2], labels)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cadec_sentiment_distribution.png'))
    plt.close()
    
    # 2. Entity type distribution
    plt.figure(figsize=(12, 6))
    entity_counts = df['entity_type'].value_counts()
    
    ax = sns.barplot(x=entity_counts.index, y=entity_counts.values)
    plt.title('Entity Type Distribution in CADEC ABSA Dataset')
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    for i, v in enumerate(entity_counts.values):
        ax.text(i, v + 5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cadec_entity_distribution.png'))
    plt.close()
    
    # 3. Token length distribution
    plt.figure(figsize=(12, 6))
    token_lengths = df['tokens'].apply(len)
    
    sns.histplot(token_lengths, bins=50, kde=True)
    plt.title('Token Length Distribution in CADEC ABSA Dataset')
    plt.xlabel('Number of Tokens')
    plt.ylabel('Frequency')
    plt.axvline(x=token_lengths.mean(), color='r', linestyle='--', 
                label=f'Mean: {token_lengths.mean():.1f}')
    plt.axvline(x=token_lengths.median(), color='g', linestyle='--', 
                label=f'Median: {token_lengths.median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cadec_token_length_distribution.png'))
    plt.close()
    
    # 4. Top aspect categories
    plt.figure(figsize=(14, 8))
    aspect_counts = df['aspect_category'].value_counts().head(15)
    
    ax = sns.barplot(x=aspect_counts.values, y=aspect_counts.index)
    plt.title('Top 15 Aspect Categories in CADEC ABSA Dataset')
    plt.xlabel('Count')
    plt.ylabel('Aspect Category')
    for i, v in enumerate(aspect_counts.values):
        ax.text(v + 3, i, str(v), va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'cadec_aspect_distribution.png'))
    plt.close()
    
    # 5. Save numeric statistics to file
    with open(os.path.join(output_folder, 'cadec_statistics.txt'), 'w') as f:
        f.write("CADEC ABSA Dataset Statistics\n")
        f.write("=" * 30 + "\n\n")
        
        f.write("1. Basic Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Total samples: {len(df)}\n")
        f.write(f"Unique reviews: {df['review_id'].nunique()}\n")
        f.write(f"Average annotations per review: {len(df) / df['review_id'].nunique():.2f}\n\n")
        
        f.write("2. Sentiment Distribution\n")
        f.write("-" * 20 + "\n")
        sentiment_dist = df['absa1'].value_counts().sort_index()
        for i, count in enumerate(sentiment_dist):
            label = labels[i]
            percentage = 100 * count / len(df)
            f.write(f"{label}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("3. Entity Type Distribution\n")
        f.write("-" * 20 + "\n")
        for entity_type, count in df['entity_type'].value_counts().items():
            percentage = 100 * count / len(df)
            f.write(f"{entity_type}: {count} ({percentage:.1f}%)\n")
        f.write("\n")
        
        f.write("4. Token Length Statistics\n")
        f.write("-" * 20 + "\n")
        f.write(f"Min tokens: {token_lengths.min()}\n")
        f.write(f"Max tokens: {token_lengths.max()}\n")
        f.write(f"Mean tokens: {token_lengths.mean():.2f}\n")
        f.write(f"Median tokens: {token_lengths.median()}\n")
        f.write("\n")
        
        f.write("5. Top 10 Aspect Categories\n")
        f.write("-" * 20 + "\n")
        for category, count in df['aspect_category'].value_counts().head(10).items():
            percentage = 100 * count / len(df)
            f.write(f"{category}: {count} ({percentage:.1f}%)\n")
    
    logger.info(f"CADEC dataset statistics saved to {output_folder}")

if __name__ == "__main__":
    # Example usage
    print("This module provides utility functions for processing CADEC data")
    print("Import and use the functions in your scripts")
