import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from sklearn.metrics import roc_auc_score, accuracy_score
from tqdm import tqdm
import os
import json
from transformers import BertModel, BertTokenizer

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from config import load_config
from src.utils.logger import AppLogger
from src.utils.checkpoint import CheckpointManager
from src.data.data_loader import RecordsDataLoader, ECGEDataset
from src.ecge.ecge import ECGE


class NoneNegClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


def parse_device(device_str: str) -> torch.device:
    device_str = device_str.strip().lower()
    
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    elif device_str.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_str)
        else:
            print(f"Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    elif device_str.isdigit():
        gpu_id = int(device_str)
        if torch.cuda.is_available():
            if gpu_id < torch.cuda.device_count():
                return torch.device(f"cuda:{gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not found (available: 0-{torch.cuda.device_count()-1}), using cuda:0")
                return torch.device("cuda:0")
        else:
            print(f"Warning: CUDA not available, falling back to CPU")
            return torch.device("cpu")
    else:
        print(f"Warning: Unknown device '{device_str}', falling back to auto detection")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ECGE50Trainer:
    DATA_SPLIT_SEED = 42
    
    def __init__(self, config_path: str = None, device: str = None):
        self.config = load_config(config_path)
        self.logger = AppLogger("ECGE_50_Trainer", level=self._get_log_level())
        train_config = self.config['training']
        self.batch_size = train_config['batch_size']
        self.epochs = train_config['epochs']
        self.learning_rate = train_config['learning_rate']
        self.hidden_dim = train_config['hidden_dim']
        self.input_dim = train_config['input_dim']
        self.layer_dim = train_config['layer_dim']
        self.pad_len = train_config['pad_len']
        self.seed = train_config['seed']
        self.data_ratio = 0.5
        if device is not None:
            self.device = parse_device(device)
        else:
            config_device = train_config.get('device', 'auto')
            self.device = parse_device(config_device)
        
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            gpu_name = torch.cuda.get_device_name(self.device)
            self.logger.info(f"GPU: {gpu_name}")
        self.logger.info(f"Training E-CGE model with {self.data_ratio*100:.0f}% of data (subset of 80%)")
        self.use_amp = train_config.get('use_amp', True) and self.device.type == 'cuda'
        self.gradient_accumulation_steps = 2
        self.num_workers = 0 if self.device.type == 'cpu' else train_config.get('num_workers', 4)
        self.pin_memory = train_config.get('pin_memory', True) and self.device.type == 'cuda'
        data_config = self.config['data']
        self.domain = data_config['domain']
        base_path = Path(data_config['base_path']) / self.domain
        
        self.records_path = base_path / data_config['records_file']
        self.problem_info_path = base_path / data_config['problem_info_file']
        self.concept2id_path = base_path / data_config['concept2id_file']
        base_model_dir = Path(self.config['model'].get('base_model_dir', './models'))
        self.model_path = base_model_dir / self.domain / "ecge_50"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Model will be saved to: {self.model_path}")
        self.kt_80_model_path = base_model_dir / self.domain / "ecge_kt_80"
        checkpoint_dir = Path(self.config['model']['checkpoint_dir']) / self.domain / "ecge_50"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))
        self._set_seed()
        self._load_data()
        self._init_model()
        self._init_optimizer()
        self.logger.info("ECGE 50% Trainer initialized")
    
    def _get_log_level(self):
        level_str = self.config['logging'].get('level', 'INFO')
        import logging
        return getattr(logging, level_str)
    
    def _set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_data(self):
        self.logger.info("Loading data...")
        data_loader = RecordsDataLoader(
            str(self.records_path),
            str(self.problem_info_path),
            str(self.concept2id_path)
        )
        bert_path = self.config['model']['bert_path']
        bert = BertModel.from_pretrained(bert_path)
        tokenizer = BertTokenizer.from_pretrained(bert_path)
        
        num_concepts = len(data_loader.concept2id)
        all_records = data_loader.load_records()
        self.logger.info(f"Loaded {len(all_records)} total records")
        indices_80_file = self.kt_80_model_path / "sampled_indices_80.json"
        if indices_80_file.exists():
            self.logger.info(f"Found 80% indices file: {indices_80_file}")
            with open(indices_80_file, 'r') as f:
                indices_80 = json.load(f)
            
            self.logger.info(f"80% indices contains {len(indices_80)} records")
            
            rng = random.Random(self.DATA_SPLIT_SEED + 1)
            n_target = int(len(all_records) * self.data_ratio)
            if n_target <= len(indices_80):
                rng.shuffle(indices_80)
                sampled_indices = indices_80[:n_target]
            else:
                sampled_indices = indices_80
                self.logger.warning(f"50% target ({n_target}) > 80% available ({len(indices_80)}), using all 80%")
        else:
            self.logger.warning(f"80% indices file not found: {indices_80_file}")
            self.logger.warning("Sampling 50% independently (may not be subset of 80%)")
            self.logger.warning("Please train KT 80% model first for proper experiment design!")
            
            rng = random.Random(self.DATA_SPLIT_SEED)
            indices = list(range(len(all_records)))
            rng.shuffle(indices)
            
            n_samples = int(len(all_records) * self.data_ratio)
            sampled_indices = indices[:n_samples]
        
        sampled_records = [all_records[i] for i in sampled_indices]
        self.logger.info(f"Sampled {len(sampled_records)} records ({self.data_ratio*100:.0f}% of total)")
        indices_file = self.model_path / "sampled_indices_50.json"
        with open(indices_file, 'w') as f:
            json.dump(sampled_indices, f)
        self.logger.info(f"Saved sampled indices to {indices_file}")
        dataset = ECGEDataset(
            sampled_records,
            data_loader,
            num_concepts=num_concepts,
            pad_len=self.pad_len,
            bert=bert,
            tokenizer=tokenizer,
            device=self.device
        )
        
        self.logger.info(f"Dataset created with {len(dataset)} sequences")
        train_split = 0.9
        train_size = int(len(dataset) * train_split)
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.logger.info(f"Train size: {train_size}, Val size: {val_size}")
        train_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        if self.num_workers > 0:
            train_loader_kwargs['persistent_workers'] = True
        
        self.train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        
        val_loader_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
        }
        if self.num_workers > 0:
            val_loader_kwargs['persistent_workers'] = True
        
        self.val_loader = DataLoader(val_dataset, **val_loader_kwargs)
        
        self.num_concepts = num_concepts
        self.data_loader = data_loader
        self.logger.info("Data loaded successfully")
    
    def _init_model(self):
        self.logger.info("Initializing model...")
        
        self.model = ECGE(
            num_concepts=self.num_concepts,
            hidden_dim=self.hidden_dim,
            input_dim=self.input_dim,
            layer_dim=self.layer_dim,
            bert_path=self.config['model']['bert_path'],
            device=str(self.device)
        )
        
        self.model.to(self.device)
        self.logger.info(f"Model initialized: {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.5
        )
        
        if self.use_amp and self.device.type == 'cuda':
            try:
                self.scaler = GradScaler('cuda')
            except TypeError:
                self.scaler = GradScaler()
        else:
            self.scaler = None
            if self.use_amp and self.device.type == 'cpu':
                self.use_amp = False
        
        self.loss_func = nn.BCELoss()
    
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        accumulation_counter = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for step, batch in enumerate(pbar):
            question_ids = batch['question_ids'].to(self.device)
            answers = batch['answers'].to(self.device)
            text_embeddings = batch['text_embeddings'].to(self.device)
            concept_embs = batch['concept_embs'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            seq_len = question_ids.size(1)
            if seq_len <= 1:
                continue
            
            input_answers = answers[:, :-1]
            input_texts = text_embeddings[:, :-1]
            input_concept_embs = concept_embs[:, :-1]
            input_mask = mask[:, :-1]
            target_question_ids = question_ids[:, 1:]
            target_answers = answers[:, 1:]
            
            if self.use_amp and self.device.type == 'cuda':
                try:
                    with autocast('cuda'):
                        preds = self._forward_batch(
                            input_answers, input_texts, input_concept_embs,
                            input_mask, target_question_ids
                        )
                    loss = self.loss_func(preds.float(), target_answers.float())
                    loss = loss / self.gradient_accumulation_steps
                except TypeError:
                    with autocast():
                        preds = self._forward_batch(
                            input_answers, input_texts, input_concept_embs,
                            input_mask, target_question_ids
                        )
                    loss = self.loss_func(preds.float(), target_answers.float())
                    loss = loss / self.gradient_accumulation_steps
            else:
                preds = self._forward_batch(
                    input_answers, input_texts, input_concept_embs,
                    input_mask, target_question_ids
                )
                loss = self.loss_func(preds, target_answers.float())
                loss = loss / self.gradient_accumulation_steps
            
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulation_counter += 1
            
            if accumulation_counter % self.gradient_accumulation_steps == 0:
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                accumulation_counter = 0
            
            if accumulation_counter == 0:
                self.model.output_layer.apply(NoneNegClipper())
            
            total_loss += loss.item() * self.gradient_accumulation_steps
            all_preds.extend(preds.detach().cpu().numpy().flatten())
            all_labels.extend(target_answers.cpu().numpy().flatten())
            
            pbar.set_postfix({
                'loss': f'{loss.item() * self.gradient_accumulation_steps:.4f}',
                'avg_loss': f'{total_loss / (step + 1):.4f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        
        return avg_loss, auc, acc
    
    def _forward_batch(self, answers, text_embeddings, concept_embs, mask, target_question_ids):
        batch_size, seq_len = answers.shape
        target_concept_ids = torch.zeros(batch_size, seq_len, dtype=torch.long).to(self.device)
        
        for b in range(batch_size):
            for t in range(seq_len):
                if mask[b, t] == 0:
                    continue
                concept_indices = torch.nonzero(concept_embs[b, t] > 0.5)
                if len(concept_indices) > 0:
                    target_concept_ids[b, t] = concept_indices[0].item()
                else:
                    target_concept_ids[b, t] = 0
        
        preds = self.model(
            text_embeddings, answers, mask=mask, target_concept_ids=target_concept_ids
        )
        
        return preds
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                question_ids = batch['question_ids'].to(self.device)
                answers = batch['answers'].to(self.device)
                text_embeddings = batch['text_embeddings'].to(self.device)
                concept_embs = batch['concept_embs'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                seq_len = question_ids.size(1)
                if seq_len <= 1:
                    continue
                
                input_answers = answers[:, :-1]
                input_texts = text_embeddings[:, :-1]
                input_concept_embs = concept_embs[:, :-1]
                input_mask = mask[:, :-1]
                target_question_ids = question_ids[:, 1:]
                target_answers = answers[:, 1:]
                
                if self.use_amp and self.device.type == 'cuda':
                    try:
                        with autocast('cuda'):
                            preds = self._forward_batch(
                                input_answers, input_texts, input_concept_embs,
                                input_mask, target_question_ids
                            )
                        loss = self.loss_func(preds.float(), target_answers.float())
                    except TypeError:
                        with autocast():
                            preds = self._forward_batch(
                                input_answers, input_texts, input_concept_embs,
                                input_mask, target_question_ids
                            )
                        loss = self.loss_func(preds.float(), target_answers.float())
                else:
                    preds = self._forward_batch(
                        input_answers, input_texts, input_concept_embs,
                        input_mask, target_question_ids
                    )
                    loss = self.loss_func(preds, target_answers.float())
                
                total_loss += loss.item()
                all_preds.extend(preds.float().detach().cpu().numpy().flatten())
                all_labels.extend(target_answers.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(self.val_loader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        acc = accuracy_score(all_labels, (np.array(all_preds) > 0.5).astype(int))
        
        return avg_loss, auc, acc
    
    def train(self, resume_from_checkpoint: str = None):
        start_epoch = 0
        best_auc = 0.0
        
        if resume_from_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            checkpoint = self.checkpoint_manager.load_checkpoint()
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch = checkpoint['epoch'] + 1
            best_auc = checkpoint.get('metrics', {}).get('best_auc', 0.0)
        
        self.logger.info("Starting training (E-CGE 50%)...")
        
        for epoch in range(start_epoch, self.epochs):
            train_loss, train_auc, train_acc = self.train_epoch(epoch)
            val_loss, val_auc, val_acc = self.validate()
            self.scheduler.step()
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            if val_auc > best_auc:
                best_auc = val_auc
                self._save_model(epoch, val_auc, is_best=True)
                self.logger.info(f"New best model saved! AUC: {best_auc:.4f}")
            
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model_state=self.model.state_dict(),
                optimizer_state=self.optimizer.state_dict(),
                metrics={'best_auc': best_auc, 'val_auc': val_auc, 'val_acc': val_acc},
                metadata={'epoch': epoch, 'data_ratio': self.data_ratio}
            )
        
        self.logger.info(f"Training completed! Best AUC: {best_auc:.4f}")
        self.logger.info(f"Model saved to: {self.model_path}")
    
    def _save_model(self, epoch: int, auc: float, is_best: bool = False):
        model_name = "Trained_E_DKT_model.pt" if is_best else f"model_epoch_{epoch}.pt"
        model_path = self.model_path / model_name
        torch.save(self.model.state_dict(), model_path)
        
        details_path = self.model_path / "Trained_E_DKT_details.txt"
        with open(details_path, 'w', encoding='utf-8') as f:
            f.write(f'Model Type: E-CGE (Recommender)\n')
            f.write(f'Data Ratio: {self.data_ratio*100:.0f}%\n')
            f.write(f'Best Val AUC: {auc:.4f}\n')
            f.write(f'Total Epochs: {self.epochs}\n')
            f.write(f'Hidden Dim: {self.hidden_dim}\n')
            f.write(f'Input Dim: {self.input_dim}\n')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train E-CGE Model (50% data)")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--gpu", type=str, default=None, 
                        help="GPU device to use. Options: "
                             "'auto' (default, auto-detect), "
                             "'cpu' (force CPU), "
                             "'cuda'/'cuda:0'/'cuda:1'/... (specific GPU), "
                             "or just GPU ID like '0', '1', '2', ...")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Training E-CGE Model (50% data) for Recommender System")
    print("=" * 60)
    
    trainer = ECGE50Trainer(config_path=args.config, device=args.gpu)
    trainer.train(resume_from_checkpoint=args.resume)
