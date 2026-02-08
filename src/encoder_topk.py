#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer


class SentenceBertEncoder:
    """SentenceBERT encoder"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"Loading SentenceBERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.entity_ids = None
        
    def encode_records(self, records: Dict[int, str]) -> np.ndarray:
        self.entity_ids = sorted(records.keys())
        texts = [records[eid] for eid in self.entity_ids]
        
        print(f"Encoding {len(texts)} records...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        return self.embeddings


class ANNTopKSearch:
    """ANN TopK search using FAISS"""
    
    def __init__(self, embeddings: np.ndarray, entity_ids: List[int]):
        self.embeddings = embeddings
        self.entity_ids = np.array(entity_ids)
        self.dim = embeddings.shape[1]
        
        print(f"Building FAISS index for {len(embeddings)} records...")
        self.index = faiss.IndexFlatIP(self.dim)
        
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype('float32'))
        
        print(f"FAISS index built successfully")
    
    def search_topk(self, entity_id: int, k: int) -> Tuple[List[int], List[float]]:
        idx = np.where(self.entity_ids == entity_id)[0][0]
        
        query_embedding = self.embeddings[idx:idx+1]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        similarities, indices = self.index.search(query_embedding.astype('float32'), k + 1)
        
        similarities = similarities[0]
        indices = indices[0]
        
        mask = indices != idx
        indices = indices[mask][:k]
        similarities = similarities[mask][:k]
        
        neighbor_ids = self.entity_ids[indices].tolist()
        similarities = similarities.tolist()
        
        return neighbor_ids, similarities
    
    def compute_similarity(self, id1: int, id2: int) -> float:
        idx1 = np.where(self.entity_ids == id1)[0][0]
        idx2 = np.where(self.entity_ids == id2)[0][0]
        
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]
        
        sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(sim)
    
    def get_all_topk(self, k: int) -> Dict[int, Tuple[List[int], List[float]]]:
        print(f"Computing TopK (k={k}) for all entities...")
        
        topk_results = {}
        for entity_id in self.entity_ids:
            neighbor_ids, similarities = self.search_topk(entity_id, k)
            topk_results[int(entity_id)] = (neighbor_ids, similarities)
        
        return topk_results

