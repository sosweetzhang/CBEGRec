from typing import List, Dict, Any
from collections import defaultdict

try:
    import numpy as np  

    _ = np.mean
except Exception:
    class _NumpyFallback:
        @staticmethod
        def mean(values):
            values = list(values)
            return sum(values) / len(values) if values else 0.0

    np = _NumpyFallback()


def compute_learning_effectiveness(initial_score: float, final_score: float, max_score: float = 1.0) -> float:
    denominator = max_score - initial_score
    if denominator <= 0:
        return 0.0
    numerator = final_score - initial_score
    ep = numerator / denominator
    return max(0.0, min(1.0, ep))


class LPRMetrics:
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.episodes = []
        self.steps_to_goal = []
        self.goal_completed = []
        self.initial_scores = []
        self.final_scores = []
        self.learning_effectiveness = []
    
    def add_episode(self,
                   steps: int,
                   goal_completed: bool,
                   initial_score: float,
                   final_score: float,
                   max_steps: int = None):
        if goal_completed:
            self.steps_to_goal.append(steps)
        else:
            self.steps_to_goal.append(max_steps + 1 if max_steps else steps + 1)
        self.goal_completed.append(goal_completed)
        self.initial_scores.append(initial_score)
        self.final_scores.append(final_score)
        ep = compute_learning_effectiveness(initial_score, final_score)
        self.learning_effectiveness.append(ep)
        
        self.episodes.append({
            'steps': steps,
            'goal_completed': goal_completed,
            'initial_score': initial_score,
            'final_score': final_score,
            'learning_effectiveness': ep
        })
    
    def compute_astg(self) -> float:
        if len(self.steps_to_goal) == 0:
            return 0.0
        return float(np.mean(self.steps_to_goal))
    
    def compute_gcr(self) -> float:
        if len(self.goal_completed) == 0:
            return 0.0
        return float(np.mean(self.goal_completed))
    
    def compute_learning_effectiveness(self) -> float:
        if len(self.learning_effectiveness) == 0:
            return 0.0
        return float(np.mean(self.learning_effectiveness))
    
    def compute_all_metrics(self) -> Dict[str, float]:
        return {
            'ASTG': self.compute_astg(),
            'GCR': self.compute_gcr(),
            'Ep': self.compute_learning_effectiveness(),
            'Avg_Initial_Score': float(np.mean(self.initial_scores)) if self.initial_scores else 0.0,
            'Avg_Final_Score': float(np.mean(self.final_scores)) if self.final_scores else 0.0,
            'Num_Episodes': len(self.episodes)
        }
    
    def get_summary(self) -> str:
        metrics = self.compute_all_metrics()
        return f"""
Evaluation Summary:
  ASTG (Average Steps to Goal): {metrics['ASTG']:.2f}
  GCR (Goal Completion Rate): {metrics['GCR']:.4f} ({metrics['GCR']*100:.2f}%)
  Ep (Learning Effectiveness): {metrics['Ep']:.4f} ({metrics['Ep']*100:.2f}%)
  Avg Initial Score: {metrics['Avg_Initial_Score']:.4f}
  Avg Final Score: {metrics['Avg_Final_Score']:.4f}
  Total Episodes: {metrics['Num_Episodes']}
"""
