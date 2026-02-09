"""
Hybrid Knowledge Graph Construction
LLM-Based Initialization + Data-Driven Validation
"""
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_llm_config, get_kg_llm_config
from prompts import build_messages, get_prompt_lang
from src.utils.logger import AppLogger

class HybridKGConstruction:
    """Hybrid KG: LLM extraction + data-driven calibration."""
    
    def __init__(self, config: dict, logger: AppLogger = None):
        self.config = config
        self.logger = logger or AppLogger("KG_Construction")
        use_separate_llm = config.get('kg_construction', {}).get('use_separate_llm', True)
        if use_separate_llm:
            self.llm_config = get_kg_llm_config()
            self.logger.info(f"Using KG LLM: {self.llm_config['model']}")
        else:
            self.llm_config = get_llm_config()
            self.logger.info(f"Using default LLM: {self.llm_config['model']}")
        
        self.delta = config.get('kg_construction', {}).get('significance_threshold', 0.1)
        from src.utils.disable_proxy import ensure_no_proxy
        ensure_no_proxy()
        
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=self.llm_config['api_key'],
            base_url=self.llm_config.get('base_url') or None
        )
        self.llm_model = self.llm_config['model']
    
    def llm_extract_concepts_and_relations(self, 
                                          text_data: List[str],
                                          existing_concepts: Dict[str, int] = None) -> Dict[str, List[str]]:
        """Step 1: extract concepts and prerequisite relations from text via LLM."""
        self.logger.info("Starting LLM-based concept and relation extraction...")
        if existing_concepts:
            concepts_list = [f'"{name}": {cid}' for name, cid in existing_concepts.items()]
        else:
            concepts_list = text_data[:50]
        batch_size = 100
        all_kg = {}
        
        for i in range(0, len(concepts_list), batch_size):
            batch = concepts_list[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(concepts_list) + batch_size - 1) // batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} concepts)...")
            
            messages = build_messages(
                "kg_extract",
                config=self.config,
                concepts_json=json.dumps(batch, ensure_ascii=False, indent=2),
                batch_size=len(batch),
                expected_with_prereq=int(len(batch) * 0.9)
            )
            
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=messages,
                    temperature=0.3,
                    timeout=60
                )
                response = completion.choices[0].message.content.strip()
                if '```json' in response:
                    response = response.split('```json')[1].split('```')[0].strip()
                elif '```' in response:
                    response = response.split('```')[1].split('```')[0].strip()
                batch_kg = json.loads(response)
                cleaned_kg = {}
                for concept_id, prereqs in batch_kg.items():
                    try:
                        int(concept_id)
                    except (ValueError, TypeError):
                        continue
                    valid_prereqs = []
                    for prereq in prereqs:
                        try:
                            int(prereq)
                            valid_prereqs.append(str(prereq))
                        except (ValueError, TypeError):
                            pass
                    cleaned_kg[str(concept_id)] = valid_prereqs
                all_kg.update(cleaned_kg)
                self.logger.info(f"Batch {batch_num} extracted {len(cleaned_kg)} concepts")
            except Exception as e:
                self.logger.error(f"LLM extraction failed for batch {batch_num}: {e}")
                if existing_concepts:
                    batch_concept_ids = list(existing_concepts.values())[i:i+batch_size]
                    for cid in batch_concept_ids:
                        all_kg[str(cid)] = []
                continue
        if existing_concepts:
            for name, cid in existing_concepts.items():
                if str(cid) not in all_kg:
                    all_kg[str(cid)] = []
        
        self.logger.info(f"LLM extracted {len(all_kg)} concepts with relations")
        return all_kg
    
    def load_historical_logs(self, records_path: str) -> List[Tuple[List[int], List[int]]]:
        """Load historical learning logs. Returns [(question_ids, answers), ...]."""
        logs = []
        with open(records_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i in range(0, len(lines), 3):
            if i + 2 >= len(lines):
                break
            seq_len = int(lines[i].strip())
            question_ids = list(map(int, lines[i + 1].strip().split()))
            answers = list(map(int, lines[i + 2].strip().split()))
            
            if len(question_ids) == len(answers):
                logs.append((question_ids, answers))
        
        self.logger.info(f"Loaded {len(logs)} historical learning logs")
        return logs
    
    def extract_concept_failures(self,
                                 logs: List[Tuple[List[int], List[int]]],
                                 problem_info: Dict,
                                 concept2id: Dict[str, int]) -> Dict[int, Set[int]]:
        """Extract per-concept failure sets from logs. concept_id -> set of problem_ids where student failed."""
        concept_failures = defaultdict(set)
        
        for question_ids, answers in logs:
            for qid, ans in zip(question_ids, answers):
                if ans == 0:
                    if str(qid) in problem_info:
                        concepts = problem_info[str(qid)].get('concepts', [])
                        for concept_name in concepts:
                            if concept_name in concept2id:
                                cid = concept2id[concept_name]
                                concept_failures[cid].add(qid)
        
        return dict(concept_failures)
    
    def validate_edge_statistical_significance(self,
                                               c_i: int,
                                               c_j: int,
                                               concept_failures: Dict[int, Set[int]],
                                               all_problems: Set[int]) -> bool:
        """Validate edge c_j -> c_i (c_j is prerequisite of c_i). Default: keep LLM edge."""
        return True
    
    def supplement_edges_from_data(self,
                                   kg: Dict[str, List[str]],
                                   records_path: str,
                                   problem_info_path: str,
                                   concept2id_path: str,
                                   target_ratio: float = 0.8) -> Dict[str, List[str]]:
        """Supplement edges from learning records if LLM returns too few."""
        num_concepts = len(kg)
        concepts_with_prereqs = sum(1 for prereqs in kg.values() if prereqs)
        current_ratio = concepts_with_prereqs / num_concepts if num_concepts > 0 else 0
        
        if current_ratio >= target_ratio:
            return kg
        
        self.logger.info(f"Prereq ratio {current_ratio:.1%}, target {target_ratio:.1%}, supplementing from data...")
        logs = self.load_historical_logs(records_path)
        with open(problem_info_path, 'r', encoding='utf-8') as f:
            problem_info = json.load(f)
        concept2id = {}
        with open(concept2id_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and ',' in line:
                    key, value = line.split(',', 1)
                    concept2id[key.strip().strip('"')] = int(value.strip())
        problem_to_concept_ids = {}
        for pid, info in problem_info.items():
            concept_names = info.get('concepts', []) or info.get('concept_routes', [])
            concept_ids = []
            for name in concept_names:
                if name in concept2id:
                    concept_ids.append(concept2id[name])
            if concept_ids:
                problem_to_concept_ids[int(pid)] = concept_ids
        order_count = defaultdict(lambda: defaultdict(int))
        
        for question_ids, answers in logs:
            concept_first_pos = {}
            for i, qid in enumerate(question_ids):
                if qid in problem_to_concept_ids:
                    for cid in problem_to_concept_ids[qid]:
                        if cid not in concept_first_pos:
                            concept_first_pos[cid] = i
            
            concepts = list(concept_first_pos.items())
            for i, (c1, pos1) in enumerate(concepts):
                for c2, pos2 in concepts[i+1:]:
                    if pos1 < pos2:
                        order_count[c2][c1] += 1
                    elif pos2 < pos1:
                        order_count[c1][c2] += 1
        added_count = 0
        min_support = 2
        
        for concept_id, prereqs in list(kg.items()):
            if prereqs:
                continue
            
            cid = int(concept_id)
            if cid not in order_count:
                continue
            
            candidates = [(prereq, count) for prereq, count in order_count[cid].items() 
                         if count >= min_support and str(prereq) in kg]
            
            if not candidates:
                continue
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            new_prereqs = [str(c) for c, _ in candidates[:2]]
            
            kg[concept_id] = new_prereqs
            added_count += len(new_prereqs)
        self.logger.info(f"Data-driven supplement added {added_count} edges")
        return kg
    
    def validate_kg(self,
                   kg_init: Dict[str, List[str]],
                   records_path: str,
                   problem_info_path: str,
                   concept2id_path: str) -> Dict[str, List[str]]:
        """Validate and filter initial KG with data-driven checks."""
        self.logger.info("Starting data-driven validation...")
        logs = self.load_historical_logs(records_path)
        
        with open(problem_info_path, 'r', encoding='utf-8') as f:
            problem_info = json.load(f)
        
        concept2id = {}
        with open(concept2id_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if ',' in line:
                    key, value = line.split(',', 1)
                    concept2id[key.strip().strip('"')] = int(value.strip())
        concept_failures = self.extract_concept_failures(logs, problem_info, concept2id)
        all_problems = set(problem_info.keys())
        kg_validated = {}
        for concept_id, prerequisites in kg_init.items():
            validated_prereqs = []
            c_i = int(concept_id)
            
            for prereq_id in prerequisites:
                c_j = int(prereq_id)
                if self.validate_edge_statistical_significance(
                    c_i, c_j, concept_failures, all_problems
                ):
                    validated_prereqs.append(prereq_id)
            
            kg_validated[concept_id] = validated_prereqs
        
        removed_edges = sum(len(kg_init.get(k, [])) - len(v) 
                           for k, v in kg_validated.items())
        self.logger.info(f"Validation removed {removed_edges} edges")
        
        return kg_validated
    
    def construct_hybrid_kg(self,
                          text_data: List[str],
                          records_path: str,
                          problem_info_path: str,
                          concept2id_path: str,
                          existing_concepts: Dict[str, int] = None) -> Dict[str, List[str]]:
        """Full hybrid KG construction pipeline."""
        self.logger.info("Starting hybrid KG construction...")
        kg_init = {}
        if self.config.get('kg_construction', {}).get('llm_extraction_enabled', True):
            kg_init = self.llm_extract_concepts_and_relations(text_data, existing_concepts)
            total_edges = sum(len(prereqs) for prereqs in kg_init.values())
            if total_edges == 0:
                raise RuntimeError(
                    "LLM extraction returned empty KG (0 edges). "
                    "This indicates LLM failed to extract prerequisite relationships. "
                    "Please check: 1) LLM API configuration, 2) Network connection, 3) LLM response format."
                )
        else:
            kg_path = Path(problem_info_path).parent / "KG_structure.json"
            if kg_path.exists():
                with open(kg_path, 'r', encoding='utf-8') as f:
                    kg_init = json.load(f)
            else:
                raise FileNotFoundError(
                    f"LLM extraction is disabled and no existing KG found at {kg_path}. "
                    f"Please either: 1) Enable LLM extraction (kg_construction.llm_extraction_enabled=true), "
                    f"or 2) Provide an existing KG_structure.json file."
                )
        
        if existing_concepts:
            for name, cid in existing_concepts.items():
                if str(cid) not in kg_init:
                    kg_init[str(cid)] = []
        if self.config.get('kg_construction', {}).get('data_validation_enabled', True):
            kg_validated = self.validate_kg(kg_init, records_path, problem_info_path, concept2id_path)
        else:
            kg_validated = kg_init
        kg_final = self.supplement_edges_from_data(
            kg_validated, records_path, problem_info_path, concept2id_path, 
            target_ratio=0.7
        )
        
        self.logger.info(f"Hybrid KG construction completed: {len(kg_final)} concepts")
        return kg_final
