"""
Checkpoint management for training and inference
"""
import json
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import torch
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import PROJECT_ROOT

class CheckpointManager:
    """Manage checkpoints for training and inference"""
    
    def __init__(self, checkpoint_dir: str, run_id: Optional[str] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        self.run_dir = self.checkpoint_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, 
                       epoch: int,
                       model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': optimizer_state,
            'metrics': metrics,
            'metadata': metadata,
            'run_id': self.run_id
        }
        
        checkpoint_path = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        latest_info = {
            'latest_epoch': epoch,
            'latest_checkpoint': str(checkpoint_path),
            'run_id': self.run_id
        }
        with open(self.run_dir / "latest_checkpoint.json", 'w') as f:
            json.dump(latest_info, f, indent=2)
    
    def load_checkpoint(self, epoch: Optional[int] = None) -> Dict[str, Any]:
        """Load checkpoint"""
        if epoch is None:
            latest_info_path = self.run_dir / "latest_checkpoint.json"
            if latest_info_path.exists():
                with open(latest_info_path, 'r') as f:
                    latest_info = json.load(f)
                checkpoint_path = Path(latest_info['latest_checkpoint'])
            else:
                raise FileNotFoundError("No checkpoint found")
        else:
            checkpoint_path = self.run_dir / f"checkpoint_epoch_{epoch}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return torch.load(checkpoint_path)
    
    def save_inference_state(self, state: Dict[str, Any], step: int):
        """Save inference state for resuming"""
        state_path = self.run_dir / f"inference_state_step_{step}.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state, f)
        latest_info = {
            'latest_step': step,
            'latest_state': str(state_path)
        }
        with open(self.run_dir / "latest_inference_state.json", 'w') as f:
            json.dump(latest_info, f, indent=2)
    
    def load_inference_state(self, step: Optional[int] = None) -> Dict[str, Any]:
        """Load inference state"""
        if step is None:
            latest_info_path = self.run_dir / "latest_inference_state.json"
            if latest_info_path.exists():
                with open(latest_info_path, 'r') as f:
                    latest_info = json.load(f)
                state_path = Path(latest_info['latest_state'])
            else:
                raise FileNotFoundError("No inference state found")
        else:
            state_path = self.run_dir / f"inference_state_step_{step}.pkl"
        
        if not state_path.exists():
            raise FileNotFoundError(f"Inference state not found: {state_path}")
        
        with open(state_path, 'rb') as f:
            return pickle.load(f)
