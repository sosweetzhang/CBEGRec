from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from transformers import BertModel, BertTokenizer  
except Exception:
    BertModel = None
    BertTokenizer = None


class ECGE(nn.Module):

    def __init__(
        self,
        num_concepts: int,
        hidden_dim: int = 199,
        input_dim: int = 128,
        layer_dim: int = 1,
        bert_path: str = "./bert-base-chinese",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_bert: bool = True,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.device = device
        self.bert_dim = 768
        self.use_bert = bool(use_bert)
        self.bert = None
        self.tokenizer = None

        if self.use_bert and BertModel is not None and BertTokenizer is not None:
            try:
                try:
                    self.bert = BertModel.from_pretrained(bert_path, use_safetensors=True)
                except Exception:
                    self.bert = BertModel.from_pretrained(bert_path)
                self.tokenizer = BertTokenizer.from_pretrained(bert_path)
                for param in self.bert.parameters():
                    param.requires_grad = False
            except Exception:
                self.bert = None
                self.tokenizer = None
                self.use_bert = False
        else:
            self.use_bert = False

        self.answer_emb = nn.Embedding(2, input_dim)
        nn.init.xavier_uniform_(self.answer_emb.weight)
        self.lstm = nn.LSTM(
            input_dim + self.bert_dim,
            hidden_dim,
            layer_dim,
            batch_first=True,
        )
        self.concept_emb = nn.Embedding(num_concepts, hidden_dim)
        nn.init.xavier_uniform_(self.concept_emb.weight)
        self.output_layer = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def _fallback_text_embedding(self, text: str) -> torch.Tensor:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [b / 255.0 for b in digest]
        repeated = (values * ((self.bert_dim // len(values)) + 1))[: self.bert_dim]
        return torch.tensor(repeated, dtype=torch.float32, device=self.device)

    def encode_exercise(self, exercise_texts: Union[List[str], torch.Tensor]) -> torch.Tensor:
        if isinstance(exercise_texts, torch.Tensor):
            return exercise_texts.to(self.device)
        if self.use_bert and self.bert is not None and self.tokenizer is not None:
            inputs = self.tokenizer(
                exercise_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.bert(**inputs)
                return outputs.last_hidden_state.mean(dim=1)
        return torch.stack([self._fallback_text_embedding(text) for text in exercise_texts], dim=0)

    def forward(
        self,
        exercise_inputs: Union[List[str], torch.Tensor],
        answers: Union[List[int], torch.Tensor],
        concept_ids: Optional[Union[List[int], torch.Tensor]] = None,
        mask: Optional[Union[List[int], torch.Tensor]] = None,
        target_concept_ids: Optional[torch.Tensor] = None,
    ) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        if isinstance(answers, torch.Tensor) and answers.dim() > 1:
            return self._forward_batch(exercise_inputs, answers, mask, target_concept_ids)
        return self._forward_single(exercise_inputs, answers, concept_ids, mask)

    def _forward_single(
        self,
        exercise_texts: List[str],
        answers: List[int],
        concept_ids: Optional[List[int]] = None,
        mask: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        seq_len = len(exercise_texts)
        with torch.no_grad():
            exercise_embs = self.encode_exercise(exercise_texts)
            answer_tensor = torch.tensor(answers, dtype=torch.long, device=self.device)
            answer_embs = self.answer_emb(answer_tensor)
            extended_inputs = []
            for i in range(seq_len):
                if answers[i] == 1:
                    extended = torch.cat([exercise_embs[i], torch.zeros(self.input_dim, device=self.device)])
                else:
                    extended = torch.cat([torch.zeros(self.bert_dim, device=self.device), answer_embs[i]])
                extended_inputs.append(extended)
            extended_inputs = torch.stack(extended_inputs).unsqueeze(0)
            if mask is None:
                mask = [1] * seq_len
            lengths = torch.tensor([sum(mask)], dtype=torch.int32)
            packed_input = pack_padded_sequence(extended_inputs, lengths, batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed_input)
            h_t = h_n[-1].squeeze(0)
            concept_ids_all = list(range(self.num_concepts)) if concept_ids is None else concept_ids
            mastery_probs = []
            uncertainties = []
            for c_id in concept_ids_all:
                k_v = self.concept_emb(torch.tensor(c_id, dtype=torch.long, device=self.device))
                combined = torch.cat([h_t, k_v])
                y_hat = self.sigmoid(self.output_layer(combined))
                mastery_probs.append(y_hat.item())
                if 0 < y_hat.item() < 1:
                    u = -(y_hat.item() * torch.log2(torch.tensor(y_hat.item())).item() + (1 - y_hat.item()) * torch.log2(torch.tensor(1 - y_hat.item())).item())
                else:
                    u = 0.0
                uncertainties.append(u)
        return {
            "hidden_state": h_t.detach(),
            "mastery_probs": torch.tensor(mastery_probs),
            "uncertainties": torch.tensor(uncertainties),
            "concept_ids": concept_ids_all,
        }

    def _forward_batch(
        self,
        text_embeddings: torch.Tensor,
        answers: torch.Tensor,
        mask: torch.Tensor,
        target_concept_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = answers.shape
        answer_embs = self.answer_emb(answers.long())
        correct_mask = (answers == 1).unsqueeze(-1)
        wrong_mask = (answers == 0).unsqueeze(-1)
        extended_inputs = torch.zeros(batch_size, seq_len, self.input_dim + self.bert_dim, device=self.device)
        if isinstance(text_embeddings, torch.Tensor):
            extended_inputs[:, :, : self.bert_dim] = text_embeddings * correct_mask.expand(-1, -1, self.bert_dim)
        extended_inputs[:, :, self.bert_dim :] = answer_embs * wrong_mask.expand(-1, -1, self.input_dim)
        lengths = mask.sum(dim=1).cpu().int()
        packed_input = pack_padded_sequence(extended_inputs, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        target_concept_embs = self.concept_emb(target_concept_ids.long())
        combined = torch.cat([lstm_out, target_concept_embs], dim=-1)
        preds = self.sigmoid(self.output_layer(combined)).squeeze(-1)
        preds = preds * mask.float()
        return preds
