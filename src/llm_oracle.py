#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from openai import AzureOpenAI, OpenAI
from data_loader import DataLoader
import tiktoken


class LLMOracle:
    
    MODEL_CONFIGS = {
        'gpt-5-mini': {
            'client_type': 'azure',
            'endpoint': '',
            'api_key': '',
            'deployment': 'gpt-5-mini',
            'api_version': '2024-12-01-preview',
            'max_completion_tokens': 16384,
            'input_price_per_million': 0.25,
            'output_price_per_million': 2.00
        }
    }
    
    def __init__(self, 
                 data_loader: DataLoader,
                 model_name: str = 'gpt-5-mini',
                 cache_dir: str = 'llm_cache',
                 use_cache: bool = True,
                 count_cache_cost: bool = False):
        self.data_loader = data_loader
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.count_cache_cost = count_cache_cost
        
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")
        self.clients = {}
        self.deployments = {}
        self.input_prices = {}
        self.output_prices = {}
        self.client_types = {}
        self.available_models = []
        
        for model, config in self.MODEL_CONFIGS.items():
            api_key = config.get('api_key', '')
            if api_key and api_key not in ['YOUR_DEEPSEEK_API_KEY', 'YOUR_META_API_KEY']:
                try:
                    client_type = config.get('client_type', 'azure')
                    if client_type == 'azure':
                        self.clients[model] = AzureOpenAI(
                            api_version=config.get('api_version', '2024-12-01-preview'),
                            azure_endpoint=config['endpoint'],
                            api_key=api_key
                        )
                    else:
                        self.clients[model] = OpenAI(
                            base_url=config['endpoint'],
                            api_key=api_key
                        )
                    self.deployments[model] = config['deployment']
                    self.client_types[model] = client_type
                    self.input_prices[model] = config['input_price_per_million'] / 1_000_000
                    self.output_prices[model] = config['output_price_per_million'] / 1_000_000
                    self.available_models.append(model)
                except Exception as e:
                    print(f"Warning: Failed to initialize model {model}: {e}")
            else:
                    print(f"Warning: Model {model} has invalid API key, skipping initialization")
        
        if not self.available_models:
            raise ValueError("No available models found. Please check API keys in MODEL_CONFIGS.")
        
        if model_name not in self.available_models:
            model_name = self.available_models[0]
            print(f"Warning: Default model not available, using {model_name} instead")
        
        self.model_name = model_name
        self.client = self.clients[model_name]
        self.deployment = self.deployments[model_name]
        self.input_price = self.input_prices[model_name]
        self.output_price = self.output_prices[model_name]
        
        print(f"Initialized LLM Oracle with model: {model_name}")
        
        self.tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        
        self.total_queries = 0
        self.correct_queries = 0
        self.query_details = []
        
        self.detailed_logs = []
        
        if self.use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            dataset_name = getattr(data_loader, 'dataset_name', 'unknown')
            self.cache_dir = cache_dir
            self.dataset_name = dataset_name
            self.cache_files = {}
            self.caches = {}
            for model in self.available_models:
                cache_file = os.path.join(cache_dir, f'{dataset_name}_{model}_cache.json')
                self.cache_files[model] = cache_file
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                            input_price = self.input_prices.get(model, 0)
                            output_price = self.output_prices.get(model, 0)
                            for key, value in cache_data.items():
                                if isinstance(value, dict) and 'cost' not in value:
                                    input_tokens = value.get('input_tokens', 0)
                                    output_tokens = value.get('output_tokens', 0)
                                    value['cost'] = (input_tokens * input_price) + (output_tokens * output_price)
                            self.caches[model] = cache_data
                    except Exception as e:
                        print(f"Warning: Failed to load cache for {model}: {e}")
                        self.caches[model] = {}
                else:
                    self.caches[model] = {}
            self.cache = self.caches.get(model_name, {})
            self.cache_file = self.cache_files.get(model_name, os.path.join(cache_dir, f'{model_name}_cache.json'))
        else:
            self.cache = {}
            self.caches = {}
            self.cache_files = {}
    
    def _load_cache_for_model(self, model_name: str) -> Dict:
        cache_file = self.cache_files.get(model_name)
        if not cache_file:
            dataset_name = getattr(self, 'dataset_name', 'unknown')
            cache_file = os.path.join(self.cache_dir, f'{dataset_name}_{model_name}_cache.json')
        
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    input_price = self.input_prices.get(model_name, 0)
                    output_price = self.output_prices.get(model_name, 0)
                    for key, value in cache_data.items():
                        if isinstance(value, dict) and 'cost' not in value:
                            input_tokens = value.get('input_tokens', 0)
                            output_tokens = value.get('output_tokens', 0)
                            value['cost'] = (input_tokens * input_price) + (output_tokens * output_price)
                    return cache_data
            except Exception as e:
                print(f"Warning: Failed to load cache for {model_name}: {e}")
                return {}
        return {}
    
    def _save_cache_for_model(self, model_name: str):
        if self.use_cache and model_name in self.caches:
            cache_file = self.cache_files.get(model_name)
            if cache_file:
                try:
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(self.caches[model_name], f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"Warning: Failed to save cache for {model_name} to {cache_file}: {e}")
    
    def _load_cache(self) -> Dict:
        return self.caches.get(self.model_name, {})
    
    def _save_cache(self):
        if self.use_cache:
            self._save_cache_for_model(self.model_name)
    
    def _get_cache_key(self, query_type: str, model_name: str, *args) -> str:
        key_str = f"{query_type}:{model_name}:{json.dumps(args, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_tokens(self, text: str) -> int:
        if self.tiktoken_encoder is not None:
            try:
                return len(self.tiktoken_encoder.encode(text))
            except Exception as e:
                pass
        return len(text) // 4
    
    def switch_model(self, model_name: str):
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available")
        self.model_name = model_name
        self.client = self.clients[model_name]
        self.deployment = self.deployments[model_name]
        self.client_type = self.client_types.get(model_name, 'azure')
        self.input_price = self.input_prices[model_name]
        self.output_price = self.output_prices[model_name]
        
        if self.use_cache:
            self.cache = self.caches.get(model_name, {})
            dataset_name = getattr(self, 'dataset_name', 'unknown')
            self.cache_file = self.cache_files.get(model_name, os.path.join(self.cache_dir, f'{dataset_name}_{model_name}_cache.json'))
    
    def get_model_cost(self, model_name: str) -> Tuple[float, float]:
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} is not available (invalid API key)")
        return (self.input_prices[model_name], self.output_prices[model_name])
    
    def is_model_available(self, model_name: str) -> bool:
        return model_name in self.available_models
    
    def get_available_models(self) -> List[str]:
        return self.available_models.copy()
    
    def _call_llm(self, prompt: str, query_type: str = "general", model_name: Optional[str] = None) -> Tuple[str, int, int, bool]:
        actual_model = model_name if model_name else self.model_name
        
        if self.use_cache:
            model_cache = self.caches.get(actual_model, {})
        else:
            model_cache = {}
        
        cache_key = self._get_cache_key(query_type, actual_model, prompt)
        is_cache_hit = cache_key in model_cache
        if is_cache_hit:
            self.cache_hits += 1
            cached_result = model_cache[cache_key]
            
            input_tokens = cached_result.get('input_tokens', self._estimate_tokens(prompt))
            output_tokens = cached_result.get('output_tokens', 10)
            
            if self.count_cache_cost:
                input_price = self.input_prices.get(actual_model, self.input_price)
                output_price = self.output_prices.get(actual_model, self.output_price)
                cost = (input_tokens * input_price) + (output_tokens * output_price)
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost += cost
            else:
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
            
            return cached_result['response'], input_tokens, output_tokens, True
        
        self.cache_misses += 1
        self.api_calls += 1
        
        actual_model = model_name if model_name else self.model_name
        config = self.MODEL_CONFIGS[actual_model]
        client = self.clients[actual_model]
        input_price = self.input_prices[actual_model]
        output_price = self.output_prices[actual_model]
        client_type = self.client_types.get(actual_model, 'azure')
        
        request_params = {
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant for entity resolution tasks.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            'model': config['deployment']
        }
        
        if client_type == 'azure':
            if 'max_completion_tokens' in config:
                request_params['max_completion_tokens'] = config['max_completion_tokens']
            if 'max_tokens' in config:
                request_params['max_tokens'] = config['max_tokens']
        else:
            if 'max_completion_tokens' in config:
                request_params['max_tokens'] = config['max_completion_tokens']
            elif 'max_tokens' in config:
                request_params['max_tokens'] = config['max_tokens']
        
        if 'temperature' in config:
            request_params['temperature'] = config['temperature']
        if 'top_p' in config:
            request_params['top_p'] = config['top_p']
        
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**request_params)
                
                result = response.choices[0].message.content
                
                input_tokens = response.usage.prompt_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'prompt_tokens') else self._estimate_tokens(prompt)
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and hasattr(response.usage, 'completion_tokens') else self._estimate_tokens(result)
                
                cost = (input_tokens * input_price) + (output_tokens * output_price)
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost += cost
                
                if self.use_cache:
                    if actual_model not in self.caches:
                        self.caches[actual_model] = {}
                    self.caches[actual_model][cache_key] = {
                        'prompt': prompt,
                        'response': result,
                        'query_type': query_type,
                        'model_name': actual_model,
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'cost': cost
                    }
                    self._save_cache_for_model(actual_model)
                
                return result, input_tokens, output_tokens, False
            except Exception as e:
                error_msg = str(e)
                is_connection_error = 'Connection' in error_msg or 'Name or service not known' in error_msg or 'timeout' in error_msg.lower()
                
                if attempt < max_retries - 1 and is_connection_error:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"  Connection error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    if is_connection_error:
                        print(f"Error calling LLM API for model {actual_model}: Connection error (after {attempt + 1} attempts)")
                        print(f"  Endpoint: {config.get('endpoint', 'N/A')}")
                        print(f"  Deployment: {config.get('deployment', 'N/A')}")
                        print(f"  Error: {error_msg[:200]}")
                    else:
                        print(f"Error calling LLM API for model {actual_model}: {error_msg[:200]}")
                    raise
    
    def reset(self):
        self.api_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_queries = 0
        self.correct_queries = 0
        self.query_details = []
    
    def query_ppl(self, entity_id: int, candidate_ids: List[int]) -> Optional[Tuple[int, List[int]]]:
        self.total_queries += 1
        
        entity_text = self.data_loader.get_entity_text(entity_id)
        candidate_texts = [self.data_loader.get_entity_text(cid) for cid in candidate_ids]
        
        prompt = f"""You are an entity resolution expert. Determine if the following entity matches any of the candidate entities.

Query Entity:
{entity_text}

Candidate Entities:
"""
        for i, ctext in enumerate(candidate_texts):
            prompt += f"{i+1}. {ctext}\n"
        
        prompt += """
Please respond with ONLY the number (1, 2, ...) of the matching candidate, or "NONE" if no match is found.
Your response should be a single number or "NONE", nothing else.
"""
        
        call_result = self._call_llm(prompt, query_type="ppl")
        if len(call_result) == 4:
            response, input_tokens, output_tokens, cache_hit = call_result
        else:
            response, input_tokens, output_tokens = call_result
            cache_hit = False
        response = response.strip().upper()
        
        matched_candidate = None
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(candidate_ids):
                matched_candidate = candidate_ids[idx]
        
        gt_matches = [cid for cid in candidate_ids if self.data_loader.is_match(entity_id, cid)]
        
        is_correct = False
        correctness_detail = ""
        if matched_candidate is not None:
            gt_match = self.data_loader.is_match(entity_id, matched_candidate)
            is_correct = gt_match
            if is_correct:
                correctness_detail = "Correct: LLM matched a ground truth entity"
            else:
                correctness_detail = f"Wrong: LLM matched {matched_candidate} but GT matches {gt_matches}"
        else:
            should_match = len(gt_matches) > 0
            is_correct = not should_match
            if is_correct:
                correctness_detail = "Correct: LLM correctly identified no match"
            else:
                correctness_detail = f"Wrong: LLM said no match but GT matches {gt_matches}"
        
        if is_correct:
            self.correct_queries += 1
        
        entity_text = self.data_loader.get_entity_text(entity_id)
        candidate_texts = {cid: self.data_loader.get_entity_text(cid) for cid in candidate_ids}
        
        self.query_details.append({
            'query_type': 'ppl',
            'entity_id': entity_id,
            'candidate_ids': candidate_ids,
            'llm_response': response,
            'matched_candidate': matched_candidate,
            'is_correct': is_correct,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': (input_tokens * self.input_price) + (output_tokens * self.output_price),
            'ground_truth_matches': gt_matches
        })
        
        detailed_log = {
            'query_id': self.total_queries,
            'timestamp': time.time(),
            'model_name': self.model_name,
            'entity_id': entity_id,
            'entity_text': entity_text,
            'candidates': [
                {
                    'id': cid,
                    'text': candidate_texts[cid],
                    'is_gt_match': cid in gt_matches
                }
                for cid in candidate_ids
            ],
            'prompt': prompt,
            'llm_response': response,
            'llm_matched_candidate_id': matched_candidate,
            'llm_matched_candidate_text': candidate_texts.get(matched_candidate) if matched_candidate else None,
            'ground_truth_matches': [
                {
                    'id': cid,
                    'text': candidate_texts[cid]
                }
                for cid in gt_matches
            ],
            'is_correct': is_correct,
            'correctness_detail': correctness_detail,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            },
            'cost': {
                'input_cost': input_tokens * self.input_price,
                'output_cost': output_tokens * self.output_price,
                'total': (input_tokens * self.input_price) + (output_tokens * self.output_price)
            },
            'cache_hit': False
        }
        self.detailed_logs.append(detailed_log)
        
        if matched_candidate is not None:
            all_matched = [cid for cid in candidate_ids 
                          if self.data_loader.is_match(entity_id, cid)]
            return matched_candidate, all_matched
        
        return None
    
    def get_stats(self) -> Dict:
        total_cache_operations = self.cache_hits + self.cache_misses
        return {
            'api_calls': self.api_calls,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'total_cost': self.total_cost,
            'total_queries': self.total_queries,
            'correct_queries': self.correct_queries,
            'query_accuracy': self.correct_queries / self.total_queries if self.total_queries > 0 else 0.0,
            'cache_hit_rate': self.cache_hits / total_cache_operations if total_cache_operations > 0 else 0.0,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'count_cache_cost': self.count_cache_cost
        }
    
    def query_ppl_with_model(self, 
                            entity_id: int, 
                            candidate_ids: List[int],
                            model_name: str) -> Optional[Tuple[int, List[int]]]:
        self.total_queries += 1
        
        entity_text = self.data_loader.get_entity_text(entity_id)
        candidate_texts = [self.data_loader.get_entity_text(cid) for cid in candidate_ids]
        
        prompt = f"""You are an entity resolution expert. Determine if the following entity matches any of the candidate entities.

Query Entity:
{entity_text}

Candidate Entities:
"""
        for i, ctext in enumerate(candidate_texts):
            prompt += f"{i+1}. {ctext}\n"
        
        prompt += """
Please respond with ONLY the number (1, 2, ...) of the matching candidate, or "NONE" if no match is found.
Your response should be a single number or "NONE", nothing else.
"""
        
        call_result = self._call_llm(prompt, query_type="ppl", model_name=model_name)
        if len(call_result) == 4:
            response, input_tokens, output_tokens, cache_hit = call_result
        else:
            response, input_tokens, output_tokens = call_result
            cache_hit = False
        response = response.strip().upper()
        
        matched_candidate = None
        if response.isdigit():
            idx = int(response) - 1
            if 0 <= idx < len(candidate_ids):
                matched_candidate = candidate_ids[idx]
        
        gt_matches = [cid for cid in candidate_ids if self.data_loader.is_match(entity_id, cid)]
        
        is_correct = False
        correctness_detail = ""
        if matched_candidate is not None:
            gt_match = self.data_loader.is_match(entity_id, matched_candidate)
            is_correct = gt_match
            if is_correct:
                correctness_detail = "Correct: LLM matched a ground truth entity"
            else:
                correctness_detail = f"Wrong: LLM matched {matched_candidate} but GT matches {gt_matches}"
        else:
            should_match = len(gt_matches) > 0
            is_correct = not should_match
            if is_correct:
                correctness_detail = "Correct: LLM correctly identified no match"
            else:
                correctness_detail = f"Wrong: LLM said no match but GT matches {gt_matches}"
        
        if is_correct:
            self.correct_queries += 1
        
        candidate_texts_dict = {cid: self.data_loader.get_entity_text(cid) for cid in candidate_ids}
        
        self.query_details.append({
            'query_type': 'ppl',
            'entity_id': entity_id,
            'candidate_ids': candidate_ids,
            'llm_response': response,
            'matched_candidate': matched_candidate,
            'is_correct': is_correct,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'model_name': model_name,
            'cost': (input_tokens * self.input_prices.get(model_name, self.input_price)) + 
                    (output_tokens * self.output_prices.get(model_name, self.output_price)),
            'ground_truth_matches': gt_matches
        })
        
        detailed_log = {
            'query_id': self.total_queries,
            'timestamp': time.time(),
            'model_name': model_name,
            'entity_id': entity_id,
            'entity_text': entity_text,
            'candidates': [
                {
                    'id': cid,
                    'text': candidate_texts_dict[cid],
                    'is_gt_match': cid in gt_matches
                }
                for cid in candidate_ids
            ],
            'prompt': prompt,
            'llm_response': response,
            'llm_matched_candidate_id': matched_candidate,
            'llm_matched_candidate_text': candidate_texts_dict.get(matched_candidate) if matched_candidate else None,
            'ground_truth_matches': [
                {
                    'id': cid,
                    'text': candidate_texts_dict[cid]
                }
                for cid in gt_matches
            ],
            'is_correct': is_correct,
            'correctness_detail': correctness_detail,
            'tokens': {
                'input': input_tokens,
                'output': output_tokens,
                'total': input_tokens + output_tokens
            },
            'cost': {
                'input_cost': input_tokens * self.input_prices.get(model_name, self.input_price),
                'output_cost': output_tokens * self.output_prices.get(model_name, self.output_price),
                'total': (input_tokens * self.input_prices.get(model_name, self.input_price)) + 
                        (output_tokens * self.output_prices.get(model_name, self.output_price))
            },
            'cache_hit': False
        }
        self.detailed_logs.append(detailed_log)
        
        if matched_candidate is not None:
            all_matched = [cid for cid in candidate_ids 
                          if self.data_loader.is_match(entity_id, cid)]
            return matched_candidate, all_matched
        
        return None
    
    def save_query_details(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_details, f, ensure_ascii=False, indent=2)
    
    def save_detailed_logs(self, filepath: str):
        log_data = {
            'summary': {
                'total_queries': self.total_queries,
                'correct_queries': self.correct_queries,
                'query_accuracy': self.correct_queries / self.total_queries if self.total_queries > 0 else 0.0,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
                'total_cost': self.total_cost,
                'model_name': self.model_name
            },
            'detailed_logs': self.detailed_logs
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

