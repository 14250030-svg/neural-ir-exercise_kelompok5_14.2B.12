"""
FiRA Judgement Aggregation Starter Code
Dosen: Zico Pratama Putra
Kelompok: 5 [Wahid Diyono, Mohamad Prastya, Wan Berry Pranata]
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
import json

def load_raw_judgements(file_path: str) -> pd.DataFrame:
    """Load raw FiRA judgements"""
    df = pd.read_csv(file_path, sep='\t')
    print(f"Loaded {len(df)} raw judgements")
    print("Columns:", df.columns.tolist())
    print(df.head())
    
    # Additional analysis
    print("\n=== Data Analysis ===")
    print("Judgement distribution:")
    print(df['judgement'].value_counts().sort_index())
    print("\nConfidence distribution:")
    print(df['confidence'].value_counts().sort_index())
    print("\nNumber of unique queries:", df['query_id'].nunique())
    print("Number of unique documents:", df['doc_id'].nunique())
    print("Number of annotators:", df['annotator_id'].nunique() if 'annotator_id' in df.columns else "N/A")
    
    return df

def simple_majority_vote(group):
    """Baseline: Simple majority vote"""
    votes = group['judgement'].value_counts()
    return votes.idxmax()

def advanced_aggregation(group):
    """
    Advanced aggregation with weighted voting based on confidence,
    outlier removal, and handling disagreement.
    """
    required_cols = ['judgement', 'confidence']
    for col in required_cols:
        if col not in group.columns:
            return round(group['judgement'].mean())
    
    # Filter low confidence judgements (confidence < 3)
    valid_group = group[group['confidence'] >= 3].copy()
    if len(valid_group) == 0:
        valid_group = group  # fallback to all if none high confidence
    
    scores = valid_group['judgement'].astype(float).values
    confidences = valid_group['confidence'].astype(float).values
    
    if len(scores) == 0:
        return round(group['judgement'].mean())
    
    # Weighted average: higher confidence gets more weight
    weights = confidences / confidences.sum() if confidences.sum() > 0 else np.ones_like(confidences)
    weighted_avg = np.average(scores, weights=weights)
    
    # Check for high disagreement (std > 1.0 and at least 3 judgements)
    if len(scores) >= 3 and np.std(scores) > 1.0:
        # Use median for high disagreement to be more robust
        final_score = np.median(scores)
    else:
        final_score = weighted_avg
    
    # Round to nearest integer (assuming relevance scores are 0-4 or similar)
    return round(final_score)

def aggregate_judgements(df: pd.DataFrame, method='advanced') -> pd.DataFrame:
    """Main aggregation function"""
    grouped = df.groupby(['query_id', 'doc_id'])
    
    aggregated = []
    for (qid, did), group in grouped:
        if method == 'majority':
            score = simple_majority_vote(group)
        else:
            score = advanced_aggregation(group)
            
        aggregated.append({
            'query_id': int(qid),
            'doc_id': str(did),
            'score': int(score),
            'num_judgements': len(group),
            'std_score': float(np.std(group['judgement']))
        })
    
    result_df = pd.DataFrame(aggregated)
    print(f"Aggregated into {len(result_df)} unique query-doc pairs")
    return result_df

def save_qrels(aggregated_df: pd.DataFrame, output_path: str):
    """Save in TREC qrels format"""
    with open(output_path, 'w') as f:
        for _, row in aggregated_df.iterrows():
            f.write(f"{row['query_id']} 0 {row['doc_id']} {row['score']}\n")
    print(f"Qrels saved to {output_path}")

# ====================== MAIN ======================
if __name__ == "__main__":
    df = load_raw_judgements("data/fira_raw_judgements.tsv")
    
    agg_df = aggregate_judgements(df, method='advanced')
    
    save_qrels(agg_df, "data/fira_aggregated.qrels")
    
    print("\n=== Aggregation Statistics ===")
    print(agg_df['score'].value_counts().sort_index())