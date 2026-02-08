#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

class ClusteringMetrics:
    """Clustering metrics calculator"""
    
    def __init__(self, ground_truth_clusters: List[List[int]], predicted_clusters: List[List[int]]):
        self.ground_truth_clusters = ground_truth_clusters
        self.predicted_clusters = predicted_clusters
        self.gt_labels, self.pred_labels, self.entity_ids = self._build_label_vectors()
    
    def _build_label_vectors(self):
        all_entities = set()
        for cluster in self.ground_truth_clusters:
            all_entities.update(cluster)
        for cluster in self.predicted_clusters:
            all_entities.update(cluster)
        
        entity_ids = sorted(all_entities)
        
        gt_labels = {}
        for label, cluster in enumerate(self.ground_truth_clusters):
            for entity_id in cluster:
                gt_labels[entity_id] = label
        
        pred_labels = {}
        for label, cluster in enumerate(self.predicted_clusters):
            for entity_id in cluster:
                pred_labels[entity_id] = label
        
        gt_vector = [gt_labels.get(eid, -1) for eid in entity_ids]
        pred_vector = [pred_labels.get(eid, -1) for eid in entity_ids]
        
        return gt_vector, pred_vector, entity_ids
    
    def compute_purity(self) -> float:
        true_label_map = {}
        for label, cluster in enumerate(self.ground_truth_clusters):
            for entity_id in cluster:
                true_label_map[entity_id] = label
        
        purity_sum = 0
        total = 0
        for pred_cluster in self.predicted_clusters:
            if len(pred_cluster) == 0:
                continue
            true_cluster_counts = defaultdict(int)
            for entity_id in pred_cluster:
                if entity_id in true_label_map:
                    true_cluster_counts[true_label_map[entity_id]] += 1
            if true_cluster_counts:
                purity_sum += max(true_cluster_counts.values())
            total += len(pred_cluster)
        
        return float(purity_sum / total) if total > 0 else 0.0
    
    def compute_inverse_purity(self) -> float:
        pred_label_map = {}
        for label, cluster in enumerate(self.predicted_clusters):
            for entity_id in cluster:
                pred_label_map[entity_id] = label
        
        inv_purity_sum = 0
        total = 0
        for true_cluster in self.ground_truth_clusters:
            if len(true_cluster) == 0:
                continue
            pred_cluster_counts = defaultdict(int)
            for entity_id in true_cluster:
                if entity_id in pred_label_map:
                    pred_cluster_counts[pred_label_map[entity_id]] += 1
            if pred_cluster_counts:
                inv_purity_sum += max(pred_cluster_counts.values())
            total += len(true_cluster)
        
        return float(inv_purity_sum / total) if total > 0 else 0.0
    
    def compute_f_measure(self) -> float:
        purity = self.compute_purity()
        inv_purity = self.compute_inverse_purity()
        if purity + inv_purity > 0:
            return float(2 * (purity * inv_purity) / (purity + inv_purity))
        return 0.0
    
    def compute_nmi(self) -> float:
        return float(normalized_mutual_info_score(self.gt_labels, self.pred_labels))
    
    def compute_all_metrics(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {
            'f_measure': self.compute_f_measure(),
            'nmi': self.compute_nmi(),
            'purity': self.compute_purity(),
            'inverse_purity': self.compute_inverse_purity()
        }

