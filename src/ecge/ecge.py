"""
Exercise-Aware Cognitive Graph Encoder (E-CGE). Single-sample inference and batch training.
"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
try:
    import huggingface_hub
    if hasattr(huggingface_hub, 'constants'):
        huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
except ImportError:
    pass

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

try:
    import transformers
    transformers.logging.set_verbosity_error()
except (ImportError, AttributeError):
    pass

class ECGE(nn.Module):
    """Exercise-Aware Cognitive Graph Encoder (text-augmented DKT)."""
    
    def __init__(self, 
                 num_concepts: int,
                 hidden_dim: int = 199,
                 input_dim: int = 128,
                 layer_dim: int = 1,
                 bert_path: str = "./bert-base-chinese",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_bert: bool = True):
        super(ECGE, self).__init__()
        
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.device = device
        self.bert_dim = 768
        self.use_bert = use_bert
        
        if not use_bert:
            raise ValueError("BERT is required but use_bert=False. Set use_bert=True and provide bert_path.")
        
        if not bert_path or not bert_path.strip():
            raise ValueError("BERT path is required but bert_path is empty. Please set bert_path in config.yaml")
        
        bert_path_str = str(bert_path).strip()
        is_local_path = os.path.exists(bert_path_str) or bert_path_str.startswith('./') or bert_path_str.startswith('../') or os.path.isabs(bert_path_str)
        if not is_local_path:
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
            print(f"Loading BERT from {bert_path} using HF mirror...")
        else:
            print(f"Loading BERT from local path: {bert_path}")
        
        try:
            try:
                self.bert = BertModel.from_pretrained(bert_path, use_safetensors=True)
            except Exception:
                self.bert = BertModel.from_pretrained(bert_path)
            self.tokenizer = BertTokenizer.from_pretrained(bert_path)
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"Successfully loaded BERT from {bert_path}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load BERT from {bert_path}: {e}. "
                f"Please check: 1) Network connection, 2) HF_ENDPOINT environment variable, "
                f"3) Model path correctness"
            ) from e
        self.answer_emb = nn.Embedding(2, input_dim)
        nn.init.xavier_uniform_(self.answer_emb.weight)
        self.lstm = nn.LSTM(
            input_dim + self.bert_dim,
            hidden_dim,
            layer_dim,
            batch_first=True
        )
        self.concept_emb = nn.Embedding(num_concepts, hidden_dim)
        nn.init.xavier_uniform_(self.concept_emb.weight)
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        
        self.to(device)
    
    def encode_exercise(self, exercise_texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        """Encode exercise texts with BERT, or return tensor as-is if already embedded."""
        if isinstance(exercise_texts, torch.Tensor):
            return exercise_texts
        
        if self.bert is None or self.tokenizer is None:
            raise RuntimeError(
                "BERT model or tokenizer is not loaded. Cannot encode exercises. "
                "Please ensure BERT is properly initialized in __init__."
            )
        
        inputs = self.tokenizer(
            exercise_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return sentence_embeddings
    
    def forward(self,
                exercise_inputs: Union[List[str], torch.Tensor],
                answers: Union[List[int], torch.Tensor],
                concept_ids: Optional[Union[List[int], torch.Tensor]] = None,
                mask: Optional[Union[List[int], torch.Tensor]] = None,
                target_concept_ids: Optional[torch.Tensor] = None) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        """Forward: single-sample returns dict (mastery_probs, etc.); batch returns tensor [B, L]."""
        if isinstance(answers, torch.Tensor) and answers.dim() > 1:
            return self._forward_batch(exercise_inputs, answers, mask, target_concept_ids)
        else:
            return self._forward_single(exercise_inputs, answers, concept_ids, mask)
    
    def _forward_single(self,
                       exercise_texts: List[str],
                       answers: List[int],
                       concept_ids: Optional[List[int]] = None,
                       mask: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """Single-sample inference (no_grad)."""
        seq_len = len(exercise_texts)
        
        with torch.no_grad():
            exercise_embs = self.encode_exercise(exercise_texts)
            answer_tensor = torch.tensor(answers, dtype=torch.long).to(self.device)
            answer_embs = self.answer_emb(answer_tensor)
            extended_inputs = []
            for i in range(seq_len):
                if answers[i] == 1:
                    extended = torch.cat([exercise_embs[i], torch.zeros(self.input_dim).to(self.device)])
                else:
                    extended = torch.cat([torch.zeros(self.bert_dim).to(self.device), answer_embs[i]])
                extended_inputs.append(extended)
            
            extended_inputs = torch.stack(extended_inputs).unsqueeze(0)
            if mask is None:
                mask = [1] * seq_len
            
            lengths = torch.tensor([sum(mask)], dtype=torch.int32)
            packed_input = pack_padded_sequence(extended_inputs, lengths, batch_first=True, enforce_sorted=False)
            lstm_out, (h_n, c_n) = self.lstm(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            
            h_t = h_n[-1].squeeze(0)
            concept_ids_all = list(range(self.num_concepts)) if concept_ids is None else concept_ids
            mastery_probs = []
            uncertainties = []
            
            for c_id in concept_ids_all:
                k_v = self.concept_emb(torch.tensor(c_id, dtype=torch.long).to(self.device))
                combined = torch.cat([h_t, k_v])
                y_hat = self.sigmoid(self.output_layer(combined))
                mastery_probs.append(y_hat.item())
                if 0 < y_hat.item() < 1:
                    u = -y_hat.item() * np.log2(y_hat.item()) - (1 - y_hat.item()) * np.log2(1 - y_hat.item())
                else:
                    u = 0.0
                uncertainties.append(u)
        
        return {
            'hidden_state': h_t.detach(),
            'mastery_probs': torch.tensor(mastery_probs),
            'uncertainties': torch.tensor(uncertainties),
            'concept_ids': concept_ids_all
        }
    
    def _forward_batch(self,
                      text_embeddings: torch.Tensor,
                      answers: torch.Tensor,
                      mask: torch.Tensor,
                      target_concept_ids: torch.Tensor) -> torch.Tensor:
        """Batch training."""
        batch_size, seq_len = answers.shape
        
        answer_embs = self.answer_emb(answers.long())
        correct_mask = (answers == 1).unsqueeze(-1)
        wrong_mask = (answers == 0).unsqueeze(-1)
        extended_inputs = torch.zeros(batch_size, seq_len, self.input_dim + self.bert_dim).to(self.device)
        extended_inputs[:, :, :self.bert_dim] = text_embeddings * correct_mask.expand(-1, -1, self.bert_dim)
        extended_inputs[:, :, self.bert_dim:] = answer_embs * wrong_mask.expand(-1, -1, self.input_dim)
        lengths = mask.sum(dim=1).cpu().int()
        packed_input = pack_padded_sequence(extended_inputs, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        target_concept_embs = self.concept_emb(target_concept_ids.long())
        combined = torch.cat([lstm_out, target_concept_embs], dim=-1)
        preds = self.sigmoid(self.output_layer(combined)).squeeze(-1)
        preds = preds * mask.float()
        
        return preds
