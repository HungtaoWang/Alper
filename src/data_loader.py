#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = SCRIPT_DIR / "data"


class DataLoader:
    """Data loader"""
    
    def __init__(self, dataset_name: str, mode: str = "whole"):
        self.dataset_name = dataset_name
        self.mode = mode
        self.dataset_dir = DATA_DIR / dataset_name
        
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        self.records = {}
        self.gt_pairs = []
        self.gt_clusters = []
        self.entity_ids = []
        
        self._load_whole_data()
    
    def _load_whole_data(self):
        dataset_file_json = self.dataset_dir / f"{self.dataset_name}.json"
        dataset_file_csv = self.dataset_dir / f"{self.dataset_name}.csv"
        
        if dataset_file_json.exists():
            df = pd.read_json(dataset_file_json, orient='records')
        elif dataset_file_csv.exists():
            df = pd.read_csv(dataset_file_csv)
        else:
            csv_files = [f for f in self.dataset_dir.glob("*.csv") if 'gt' not in f.name.lower()]
            json_files = [f for f in self.dataset_dir.glob("*.json")]
            if csv_files:
                df = pd.read_csv(csv_files[0])
            elif json_files:
                df = pd.read_json(json_files[0], orient='records')
            else:
                raise FileNotFoundError(f"Dataset file not found in {self.dataset_dir}")
        
        if 'id' not in df.columns:
            raise ValueError(f"Dataset file missing 'id' column")
        
        text_col = None
        for col in ['text', 'content', 'record', 'name', 'title']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            text_col = 'text'
            df[text_col] = df.apply(lambda row: ' '.join([str(v) for v in row.values if pd.notna(v)]), axis=1)
        
        for _, row in df.iterrows():
            entity_id = int(row['id'])
            text = str(row[text_col])
            self.records[entity_id] = text
        
        gt_file = self.dataset_dir / "gt.csv"
        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        try:
                            id1 = int(row[0].strip())
                            id2 = int(row[1].strip())
                            if id1 in self.records and id2 in self.records:
                                self.gt_pairs.append((id1, id2))
                        except ValueError:
                            continue
        
        self.entity_ids = sorted(self.records.keys())
        self.gt_clusters = self._build_clusters_from_pairs(self.gt_pairs, self.entity_ids)
    
    def _build_clusters_from_pairs(self, pairs: List[Tuple[int, int]], all_entity_ids: List[int]) -> List[List[int]]:
        graph = defaultdict(set)
        entities_in_pairs = set()
        
        for id1, id2 in pairs:
            graph[id1].add(id2)
            graph[id2].add(id1)
            entities_in_pairs.add(id1)
            entities_in_pairs.add(id2)
        
        visited = set()
        clusters = []
        
        def dfs(node: int, component: List[int]):
            visited.add(node)
            component.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        for entity in entities_in_pairs:
            if entity not in visited:
                component = []
                dfs(entity, component)
                if component:
                    clusters.append(sorted(component))
        
        all_entity_ids_set = set(all_entity_ids)
        singleton_entities = all_entity_ids_set - entities_in_pairs
        for singleton_id in sorted(singleton_entities):
            clusters.append([singleton_id])
        
        return clusters
    
    def get_entity_text(self, entity_id: int) -> str:
        if entity_id in self.records:
            return self.records[entity_id]
        return ""
    
    def is_match(self, id1: int, id2: int) -> bool:
        return (id1, id2) in self.gt_pairs or (id2, id1) in self.gt_pairs
    
    def get_ground_truth_label(self, entity_id: int) -> Optional[int]:
        for cluster_idx, cluster in enumerate(self.gt_clusters):
            if entity_id in cluster:
                return cluster_idx
        return None
    
    def get_all_records(self) -> Dict[int, str]:
        return self.records
    
    def get_ground_truth_clusters(self) -> List[List[int]]:
        return self.gt_clusters
    
    def get_entity_ids(self) -> List[int]:
        return self.entity_ids


def get_all_datasets() -> List[str]:
    datasets = []
    for dataset_dir in DATA_DIR.iterdir():
        if dataset_dir.is_dir() and not dataset_dir.name.startswith('_'):
            if (dataset_dir / f"{dataset_dir.name}.json").exists() or \
               (dataset_dir / f"{dataset_dir.name}.csv").exists():
                datasets.append(dataset_dir.name)
            else:
                csv_files = [f for f in dataset_dir.glob("*.csv") if 'gt' not in f.name.lower()]
                json_files = [f for f in dataset_dir.glob("*.json")]
                if csv_files or json_files:
                    datasets.append(dataset_dir.name)
    datasets.sort()
    return datasets
