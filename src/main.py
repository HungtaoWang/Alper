#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import time
import json
from typing import Dict, List
from datetime import datetime

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, ValueError):
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np

from encoder_topk import SentenceBertEncoder, ANNTopKSearch
from data_loader import DataLoader, get_all_datasets
from metrics import ClusteringMetrics
from llm_oracle import LLMOracle
from alper_framework import AlperFramework


def Alper(
    dataset_name: str,
    budget: float = 1000.0,
    model_name: str = "gpt-5-mini",
    k: int = 15,
    top_m: int = 5,
    alpha: float = 0.8,
    theta: float = 0.6,
    xi_llm: float = 0.95,
    L: float = 0.6,
    U: float = 0.95,
    cache_dir: str = "llm_cache",
    output_dir: str = "ceer_results",
    count_cache_cost: bool = False,
):
    """
    Run Alper Framework
    """
    print(f"\n{'='*100}")
    print(f"Alper Framework: {dataset_name}")
    print(f"Budget: {budget} USD")
    print(f"Model: {model_name}")
    print(f"{'='*100}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Loading dataset...")
    data_loader = DataLoader(dataset_name, mode="whole")
    records = data_loader.get_all_records()
    gt_clusters = data_loader.get_ground_truth_clusters()
    entity_ids = data_loader.get_entity_ids()
    
    print(f"Loaded {len(records)} records, {len(gt_clusters)} ground truth clusters")
    
    print("\nEncoding records...")
    encoder = SentenceBertEncoder()
    embeddings = encoder.encode_records(records)
    topk_search = ANNTopKSearch(embeddings, entity_ids)
    
    if model_name not in LLMOracle.MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(LLMOracle.MODEL_CONFIGS.keys())}")
    model_cfg = LLMOracle.MODEL_CONFIGS[model_name]
    input_price_per_m = model_cfg.get("input_price_per_million", 0.25)
    output_price_per_m = model_cfg.get("output_price_per_million", 2.00)
    input_price = input_price_per_m / 1_000_000.0
    output_price = output_price_per_m / 1_000_000.0
    
    llm_oracle = LLMOracle(
        data_loader=data_loader,
        model_name=model_name,
        cache_dir=cache_dir,
        use_cache=True,
        count_cache_cost=count_cache_cost,
    )
    
    alper = AlperFramework(
        records=records,
        data_loader=data_loader,
        topk_search=topk_search,
        llm_oracle=llm_oracle,
        budget=float(budget),
        k=k,
        max_iterations=50,
        top_m=top_m,
        alpha=alpha,
        theta=theta,
        xi_llm=xi_llm,
        L=L,
        U=U,
        input_price=input_price,
        output_price=output_price,
    )
    
    start_time = time.time()
    clusters, queries_used, stats = alper.run()
    elapsed_time = time.time() - start_time
    
    oracle_stats = llm_oracle.get_stats()
    stats['llm_query_accuracy'] = oracle_stats['query_accuracy']
    stats['cache_hit_rate'] = oracle_stats['cache_hit_rate']
    stats['input_tokens'] = oracle_stats['total_input_tokens']
    stats['output_tokens'] = oracle_stats['total_output_tokens']
    stats['total_tokens'] = oracle_stats['total_input_tokens'] + oracle_stats['total_output_tokens']
    
    print("\nComputing metrics...")
    metrics_calc = ClusteringMetrics(gt_clusters, clusters)
    metrics = metrics_calc.compute_all_metrics()
    
    result = {
        'dataset': dataset_name,
        'mode': 'alper',
        'budget': budget,
        'models': [model_name],
        'parameters': {
            'k': k,
            'top_m': top_m,
            'alpha': alpha,
            'theta': theta,
            'xi_llm': xi_llm,
            'L': L,
            'U': U,
        },
        'results': {
            'f_measure': metrics['f_measure'],
            'nmi': metrics['nmi'],
            'num_clusters': len(clusters),
            'gt_clusters': len(gt_clusters),
            'queries_used': queries_used,
            'total_cost': stats['total_cost'],
            'budget_utilization': stats['budget_utilization'],
            'time': elapsed_time,
        },
        'timestamp': timestamp
    }
    
    if 'llm_query_accuracy' in stats:
        result['results']['llm_query_accuracy'] = stats['llm_query_accuracy']
        result['results']['cache_hit_rate'] = stats['cache_hit_rate']
        oracle_stats_detail = llm_oracle.get_stats()
        result['results']['correct_queries'] = oracle_stats_detail.get('correct_queries', 0)
        result['results']['total_llm_queries'] = oracle_stats_detail.get('total_queries', 0)
        result['results']['api_calls'] = oracle_stats_detail.get('api_calls', 0)
        result['results']['cache_hits'] = oracle_stats_detail.get('cache_hits', 0)
        result['results']['cache_misses'] = oracle_stats_detail.get('cache_misses', 0)
        result['results']['actual_model_used'] = llm_oracle.model_name
        result['results']['requested_model'] = model_name
    
    if 'input_tokens' in stats:
        result['results']['input_tokens'] = stats['input_tokens']
        result['results']['output_tokens'] = stats['output_tokens']
        result['results']['total_tokens'] = stats['total_tokens']
        result['results']['avg_tokens_per_query'] = (stats['total_tokens'] / queries_used) if queries_used > 0 else 0
    
    result_file = os.path.join(output_dir, f"{dataset_name}_alper_{timestamp}.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    detailed_log_file = os.path.join(output_dir, f"{dataset_name}_alper_{timestamp}_logs.json")
    llm_oracle.save_detailed_logs(detailed_log_file)
    print(f"Detailed LLM logs saved to: {detailed_log_file}")
    
    print("\n" + "="*100)
    print("Results:")
    print("="*100)
    print(f"F-measure: {metrics['f_measure']:.4f}")
    print(f"NMI: {metrics['nmi']:.4f}")
    print(f"Clusters: {len(clusters)} (GT: {len(gt_clusters)})")
    print(f"Queries: {queries_used}")
    print(f"Cost: {stats['total_cost']:.4f} USD")
    print(f"Budget Utilization: {stats['budget_utilization']:.1f}% ({stats['total_cost']:.4f}/{budget:.4f} USD)")
    
    if 'llm_query_accuracy' in stats:
        oracle_stats_detail = llm_oracle.get_stats()
        print(f"\n{'='*100}")
        print("LLM Oracle Statistics (vs Ground Truth)")
        print(f"{'='*100}")
        print(f"Query Accuracy: {stats['llm_query_accuracy']:.4f} ({stats['llm_query_accuracy']*100:.2f}%)")
        print(f"  - Correct Queries: {oracle_stats_detail.get('correct_queries', 0)}")
        print(f"  - Total Queries: {oracle_stats_detail.get('total_queries', 0)}")
        print(f"  - Incorrect Queries: {oracle_stats_detail.get('total_queries', 0) - oracle_stats_detail.get('correct_queries', 0)}")
        print(f"\nCache Performance:")
        print(f"  Cache Hit Rate: {stats['cache_hit_rate']:.4f} ({stats['cache_hit_rate']*100:.2f}%)")
        print(f"  - Cache Hits: {oracle_stats_detail.get('cache_hits', 0)}")
        print(f"  - Cache Misses: {oracle_stats_detail.get('cache_misses', 0)}")
        print(f"  - API Calls: {oracle_stats_detail.get('api_calls', 0)}")
    
    if 'input_tokens' in stats:
        print(f"\nToken Usage (estimated with tiktoken):")
        print(f"  Input Tokens: {stats['input_tokens']:,}")
        print(f"  Output Tokens: {stats['output_tokens']:,}")
        print(f"  Total Tokens: {stats['total_tokens']:,}")
        if queries_used > 0:
            print(f"  Avg Tokens per Query: {stats['total_tokens'] / queries_used:,.1f}")
        
        actual_model = llm_oracle.model_name
        model_config = llm_oracle.MODEL_CONFIGS.get(actual_model, {})
        input_price_per_m = model_config.get('input_price_per_million', 0.15)
        output_price_per_m = model_config.get('output_price_per_million', 0.60)
        input_cost = (stats['input_tokens'] / 1_000_000) * input_price_per_m
        output_cost = (stats['output_tokens'] / 1_000_000) * output_price_per_m
        print(f"\nCost Breakdown (Model: {actual_model}):")
        print(f"  Input Cost: ${input_cost:.4f} (${input_price_per_m:.2f}/1M tokens)")
        print(f"  Output Cost: ${output_cost:.4f} (${output_price_per_m:.2f}/1M tokens)")
        print(f"  Total Cost: ${stats['total_cost']:.4f}")
        calculated_total = input_cost + output_cost
        if abs(calculated_total - stats['total_cost']) > 0.0001:
            print(f"  [Warning] Cost mismatch: Calculated ${calculated_total:.4f} vs Actual ${stats['total_cost']:.4f}")
    
    print(f"Time: {elapsed_time:.2f}s")
    print(f"\nResults saved to: {result_file}")
    print("="*100 + "\n")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Alper Framework')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name')
    parser.add_argument('--budget', type=float, default=1000.0,
                       help='Total budget (USD, default: 1000.0)')
    parser.add_argument('--model', type=str, default='gpt-5-mini',
                       help='LLM model name (default: gpt-5-mini)')
    parser.add_argument('--k', type=int, default=15,
                       help='K-NN neighbors (default: 15)')
    parser.add_argument('--top-m', type=int, default=5,
                       help='Top-M candidates (default: 5)')
    parser.add_argument('--alpha', type=float, default=0.8,
                       help='Confidence in similarity (default: 0.8)')
    parser.add_argument('--theta', type=float, default=0.6,
                       help='Label confidence threshold for graph propagation only (default: 0.6)')
    parser.add_argument('--xi-llm', type=float, default=0.95,
                       help='Expected LLM accuracy (default: 0.95)')
    parser.add_argument('--L', type=float, default=0.6,
                       help='Global efficiency lower bound L (default: 0.6)')
    parser.add_argument('--U', type=float, default=0.95,
                       help='Global efficiency upper bound U (default: 0.95)')
    parser.add_argument('--cache-dir', type=str, default='llm_cache',
                       help='LLM cache directory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--count-cache-cost', action='store_true',
                       help='Count cache hit cost')
    parser.add_argument('--list', action='store_true',
                       help='List all available datasets')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable datasets:")
        print("="*80)
        datasets = get_all_datasets()
        if not datasets:
            print("No datasets found!")
        else:
            for name in datasets:
                print(f"  {name}")
        print("="*80)
        sys.exit(0)
    
    try:
        result = Alper(
            dataset_name=args.dataset,
            budget=args.budget,
            model_name=args.model,
            k=args.k,
            top_m=args.top_m,
            alpha=args.alpha,
            theta=args.theta,
            xi_llm=args.xi_llm,
            L=args.L,
            U=args.U,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            count_cache_cost=args.count_cache_cost,
        )
        print("Completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

