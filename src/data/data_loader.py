import json
import copy
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

try:
    import numpy as np  

    _ = np.random.RandomState
except Exception:
    class _PythonRandomState:
        def __init__(self, seed=None):
            self._rng = random.Random(seed)

        def randint(self, low, high=None):
            if high is None:
                return self._rng.randrange(low)
            return self._rng.randrange(low, high)

        def choice(self, values):
            values = list(values)
            return values[self._rng.randrange(len(values))]

    class _RandomNamespace:
        RandomState = _PythonRandomState

    class _NumpyFallback:
        random = _RandomNamespace()

    np = _NumpyFallback()

try:
    import torch  
    from torch.utils.data import Dataset, DataLoader  
except Exception:
    torch = None

    class Dataset:
        pass

    DataLoader = None

class RecordsDataLoader:
    
    def __init__(self, records_path: str, problem_info_path: str, concept2id_path: str):
        self.records_path = records_path
        self.problem_info_path = problem_info_path
        self.concept2id_path = concept2id_path
        with open(problem_info_path, 'r', encoding='utf-8') as f:
            self.problem_info = json.load(f)
        self.concept2id = {}
        with open(concept2id_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ',' in line:
                    key, value = line.strip().split(',', 1)
                    self.concept2id[key.strip('"')] = int(value)
        
        self.id2concept = {v: k for k, v in self.concept2id.items()}
    
    def load_records(self) -> List[Tuple[int, List[int], List[int]]]:
        records = []
        with open(self.records_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if i + 2 >= len(lines):
                break
            
            try:
                seq_len = int(lines[i].strip())
                question_ids = list(map(int, lines[i + 1].strip().split()))
                answers = list(map(int, lines[i + 2].strip().split()))
                if len(question_ids) == len(answers) and len(question_ids) == seq_len:
                    records.append((seq_len, question_ids, answers))
                else:
                    print(f"Warning: Mismatch in record at line {i}, skipping")
                
                i += 3
            except (ValueError, IndexError) as e:
                print(f"Error parsing record at line {i}: {e}")
                i += 1
        
        return records
    
    def get_problem_texts(self, question_ids: List[int]) -> List[str]:
        texts = []
        for qid in question_ids:
            if str(qid) in self.problem_info:
                texts.append(self.problem_info[str(qid)].get('content', ''))
            else:
                texts.append('')
        return texts
    
    def get_concept_embeddings(self, question_ids: List[int], num_concepts: int) -> List[List[float]]:
        embeddings = []
        for qid in question_ids:
            emb = [0.0] * num_concepts
            if str(qid) in self.problem_info:
                concepts = self.problem_info[str(qid)].get('concepts', [])
                for concept_name in concepts:
                    if concept_name in self.concept2id:
                        cid = self.concept2id[concept_name]
                        if cid < num_concepts:
                            emb[cid] = 1.0
            embeddings.append(emb)
        return embeddings


class Learner:
    
    def __init__(self, initial_log: List, learning_target: set, _id=None, seed=None):
        self._target = learning_target
        self._logs = initial_log
        self._state = {}
        if seed is not None:
            self.random_state = np.random.RandomState(seed)
        else:
            self.random_state = np.random.RandomState()
    
    def update_logs(self, logs):
        self._logs = logs
    
    @property
    def profile(self):
        return {
            "id": getattr(self, 'id', None),
            "logs": self._logs,
            "target": self.target
        }
    
    def learn(self, learning_item, score):
        self._logs[0].append(0)
        self._logs[1].append(learning_item)
        self._logs[2].append(score)
    
    @property
    def state(self):
        return self._state
    
    def response(self, test_item) -> float:
        return self._state.get(str(test_item), 0.0)
    
    @property
    def target(self):
        return self._target


class KESPhysicsLearnerGroup:
    
    def __init__(self, dataRec_path: str, seed: int = 2024):
        self.data_path = dataRec_path
        self.random_state = np.random.RandomState(seed)
        problem_info_path = dataRec_path.replace("data_rec.txt", "problem_info.json")
        if os.path.exists(problem_info_path):
            with open(problem_info_path, 'r', encoding='utf-8') as f:
                self.p_info = json.load(f)
        else:
            self.p_info = {}
    
    def __next__(self):
        records = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            if i + 2 >= len(lines):
                break
            try:
                seq_len = int(lines[i].strip())
                question_ids = list(map(int, lines[i + 1].strip().split()))
                answers = list(map(int, lines[i + 2].strip().split()))
                if len(question_ids) == len(answers) == seq_len:
                    records.append((seq_len, question_ids, answers))
                i += 3
            except (ValueError, IndexError):
                i += 1
        
        if not records:
            raise StopIteration("No valid records found")
        index = self.random_state.randint(len(records))
        seq_len, stu_qdata, stu_ansdata = records[index]
        stu_qtdata = [self.p_info.get(str(qid), {}).get("content", "") for qid in stu_qdata]
        initial_len = int(0.6 * len(stu_qdata))
        session = [stu_qdata[:initial_len], stu_qtdata[:initial_len], stu_ansdata[:initial_len]]
        target_start_idx = int(0.8 * len(stu_qdata))
        if target_start_idx < len(stu_qdata):
            available_targets = stu_qdata[target_start_idx:]
            learning_targets = {self.random_state.choice(available_targets)} if available_targets else {stu_qdata[-1]}
        else:
            learning_targets = {stu_qdata[-1]} if stu_qdata else set()
        
        return Learner(copy.deepcopy(session), learning_targets, seed=self.random_state.randint(10000))


class ECGEDataset(Dataset):
    
    def __init__(self,
                 records: List[Tuple[int, List[int], List[int]]],
                 data_loader: RecordsDataLoader,
                 num_concepts: int,
                 pad_len: int = 20,
                 bert: Optional[object] = None,
                 tokenizer: Optional[object] = None,
                 device: str = "cpu"):
        if torch is None:
            raise RuntimeError("ECGEDataset requires torch. Install torch before training E-CGE.")
        self.records = records
        self.data_loader = data_loader
        self.num_concepts = num_concepts
        self.pad_len = pad_len
        self.device = device
        self.bert = bert
        self.tokenizer = tokenizer
        if self.bert is not None:
            self.bert = self.bert.cpu()
            self.bert.eval()
        self.sequences = self._process_records()
    
    def _process_records(self) -> List[Dict]:
        sequences = []
        
        for seq_len, question_ids, answers in tqdm(self.records, desc="Processing records"):
            question_texts = self.data_loader.get_problem_texts(question_ids)
            concept_embs = self.data_loader.get_concept_embeddings(question_ids, self.num_concepts)
            idx = 0
            while idx < seq_len:
                end_idx = min(idx + self.pad_len, seq_len)
                
                seq_qids = question_ids[idx:end_idx]
                seq_texts = question_texts[idx:end_idx]
                seq_answers = answers[idx:end_idx]
                seq_concept_embs = concept_embs[idx:end_idx]
                if len(seq_qids) < self.pad_len:
                    pad_len_needed = self.pad_len - len(seq_qids)
                    seq_qids = seq_qids + [0] * pad_len_needed
                    seq_texts = seq_texts + [''] * pad_len_needed
                    seq_answers = seq_answers + [0] * pad_len_needed
                    seq_concept_embs = seq_concept_embs + [[0.0] * self.num_concepts] * pad_len_needed
                
                mask = [1] * (end_idx - idx) + [0] * (self.pad_len - (end_idx - idx))
                
                sequences.append({
                    'question_ids': seq_qids,
                    'question_texts': seq_texts,
                    'answers': seq_answers,
                    'concept_embs': seq_concept_embs,
                    'mask': mask
                })
                
                idx = end_idx
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if torch is None:
            raise RuntimeError("ECGEDataset requires torch. Install torch before training E-CGE.")
        seq = self.sequences[idx]
        if self.bert is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                seq['question_texts'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = self.bert(**inputs)
                text_embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            text_embeddings = seq['question_texts']
        return {
            'question_ids': torch.tensor(seq['question_ids'], dtype=torch.long),
            'answers': torch.tensor(seq['answers'], dtype=torch.long),
            'text_embeddings': text_embeddings,
            'concept_embs': torch.tensor(seq['concept_embs'], dtype=torch.float),
            'mask': torch.tensor(seq['mask'], dtype=torch.long)
        }
