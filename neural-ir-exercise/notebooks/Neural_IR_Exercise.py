"""
FiRA Full Pipeline: Judgement Aggregation + Neural Re-Ranking + Extractive QA
Dosen: Zico Pratama Putra
Kelompok: 5 [Mohamad Prastya - Wahid Diyono - Wan Berry Pranata]
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np

# Import modul-modul
from judgement_aggregation import load_raw_judgements, aggregate_judgements, save_qrels
from bert_cross_encoder import BERTCrossEncoder
from extractive_qa import ExtractiveQA, evaluate_qa

# ====================== CONFIG ======================
DATA_DIR = "data"
RAW_JUDGEMENTS = f"{DATA_DIR}/fira_raw_judgements.tsv"
TUPLES_FILE = f"{DATA_DIR}/fira-2022.tuples.tsv"
BASELINE_QRELS = f"{DATA_DIR}/fira-2022.baseline-qrels.tsv"
AGG_QRELS = f"{DATA_DIR}/fira_aggregated.qrels"

# ====================== PART 1: Judgement Aggregation ======================
def run_aggregation():
    print("\n" + "="*60)
    print("PART 1: JUDGEMENT AGGREGATION")
    print("="*60)
    
    df = load_raw_judgements(RAW_JUDGEMENTS)
    agg_df = aggregate_judgements(df, method='advanced')
    save_qrels(agg_df, AGG_QRELS)
    return agg_df

# ====================== PART 2: Re-Ranking ======================
def load_tuples(file_path: str):
    """Load tuples untuk re-ranking"""
    df = pd.read_csv(file_path, sep='\t', header=None, 
                     names=['query_id', 'doc_id', 'query', 'passage'])
    print(f"Loaded {len(df)} query-passage pairs")
    return df

def load_qrels(qrels_path: str):
    """Load qrels menjadi dictionary"""
    qrels = {}
    with open(qrels_path, 'r') as f:
        for line in f:
            if line.strip():
                qid, _, did, rel = line.strip().split()
                qrels[(int(qid), str(did))] = int(rel)
    return qrels

def run_reranking(reranker, tuples_df, qrels_dict, top_k=10):
    """Re-rank dan evaluasi"""
    print("\n" + "="*60)
    print("PART 2: NEURAL RE-RANKING")
    print("="*60)
    
    from collections import defaultdict
    query_groups = defaultdict(list)
    query_text = {}
    
    for _, row in tuples_df.iterrows():
        qid = int(row['query_id'])
        query_groups[qid].append({
            'doc_id': row['doc_id'],
            'passage': row['passage']
        })
        query_text[qid] = row['query']
    
    results = []
    for qid, items in tqdm(query_groups.items(), desc="Processing queries"):
        passages = [item['passage'] for item in items]
        doc_ids = [item['doc_id'] for item in items]
        
        ranked_indices, scores = reranker.re_rank(query_text[qid], passages)
        
        # Ambil top-k
        for rank, idx in enumerate(ranked_indices[:top_k]):
            did = doc_ids[idx]
            rel = qrels_dict.get((qid, did), 0)
            results.append({
                'query_id': qid,
                'doc_id': did,
                'rank': rank + 1,
                'score': scores[idx],
                'relevance': rel,
                'query': query_text[qid],
                'passage': passages[idx]
            })
    
    return pd.DataFrame(results)

# ====================== PART 3: Extractive QA ======================
def run_qa_evaluation(reranker_results: pd.DataFrame, qa_system):
    print("\n" + "="*60)
    print("PART 3: EXTRACTIVE QUESTION ANSWERING")
    print("="*60)
    
    # Ambil top-1 passage untuk setiap query
    top_passages = reranker_results[reranker_results['rank'] == 1]
    
    questions = []
    contexts = []
    gold_answers = []  # TODO: Ganti dengan gold answer asli jika ada
    
    for _, row in top_passages.iterrows():
        questions.append(row['query'])
        contexts.append(row['passage'])
        gold_answers.append("")  # Placeholder - ganti dengan gold answer jika tersedia
    
    # Jalankan QA
    predictions = qa_system.batch_answer(questions, contexts)
    
    # Evaluasi
    eval_result = evaluate_qa(predictions, gold_answers)
    print("\n=== QA Evaluation Results ===")
    print(f"Exact Match : {eval_result['Exact_Match']:.2f}%")
    print(f"F1 Score    : {eval_result['F1_Score']:.2f}%")
    
    return predictions

def evaluate_reranking_metrics(results_df: pd.DataFrame, top_k: int = 10):
    """
    Menghitung MRR@k, NDCG@k, dan Precision@k dari hasil re-ranking.
    Asumsi: relevansi > 0 dianggap relevan (untuk MRR dan Precision).
    """
    print(f"\nMenghitung Metrik Evaluasi (Top-{top_k})...")
    
    def calculate_query_metrics(group):
        # Pastikan data diurutkan berdasarkan rank
        group = group.sort_values('rank').head(top_k)
        rels = group['relevance'].values
        
        # 1. Precision@k
        # Mengubah nilai relevansi menjadi biner (1 jika relevan, 0 jika tidak)
        binary_rels = (rels > 0).astype(int)
        precision = np.sum(binary_rels) / top_k
        
        # 2. MRR@k
        mrr = 0.0
        relevant_ranks = np.where(binary_rels == 1)[0]
        if len(relevant_ranks) > 0:
            first_rel_rank = relevant_ranks[0] + 1 # Rank dimulai dari 1
            mrr = 1.0 / first_rel_rank
            
        # 3. NDCG@k
        # Discounted Cumulative Gain (DCG)
        discounts = np.log2(np.arange(2, len(rels) + 2))
        dcg = np.sum(rels / discounts)
        
        # Ideal DCG (IDCG) - dihitung dari dokumen terbaik yang diretrieve
        ideal_rels = np.sort(rels)[::-1]
        idcg = np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2)))
        
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        
        return pd.Series({'MRR': mrr, 'NDCG': ndcg, 'Precision': precision})

    # Hitung metrik per query, lalu ambil rata-ratanya
    metrics_per_query = results_df.groupby('query_id').apply(calculate_query_metrics)
    mean_metrics = metrics_per_query.mean()
    
    print("-" * 30)
    print(f"MRR@{top_k:<8} : {mean_metrics['MRR']:.4f}")
    print(f"NDCG@{top_k:<7} : {mean_metrics['NDCG']:.4f}")
    print(f"Precision@{top_k:<2}: {mean_metrics['Precision']:.4f}")
    print("-" * 30)
    
    return mean_metrics

# ====================== MAIN PIPELINE ======================
if __name__ == "__main__":
    print("🚀 Memulai Full FiRA Pipeline...\n")
    
    # Part 1
    run_aggregation()
    
    # Inisialisasi model
    print("\nLoading models...")
    reranker = BERTCrossEncoder()
    qa_system = ExtractiveQA()
    
    # Load data untuk re-ranking
    tuples_df = load_tuples(TUPLES_FILE)
    
    # Load qrels
    baseline_qrels = load_qrels(BASELINE_QRELS)
    agg_qrels = load_qrels(AGG_QRELS)
    
    # Part 2 - Evaluasi dengan 2 jenis qrels
    print("\nEvaluating with Baseline Qrels...")
    baseline_results = run_reranking(reranker, tuples_df, baseline_qrels, top_k=10)
    eval_baseline = evaluate_reranking_metrics(baseline_results, top_k=10)
    
    print("\nEvaluating with Aggregated Qrels (Custom)...")
    agg_results = run_reranking(reranker, tuples_df, agg_qrels, top_k=10)
    eval_agg = evaluate_reranking_metrics(agg_results, top_k=10)
    
    # Part 3 - QA pada top passage
    print("\nRunning Extractive QA on top-ranked passages...")
    qa_predictions = run_qa_evaluation(agg_results, qa_system)
    
    print("\n✅ Pipeline selesai dijalankan!")