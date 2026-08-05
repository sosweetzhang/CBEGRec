import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class OutputManager:
    
    def __init__(self, output_dir: str, domain: Optional[str] = None, run_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.domain = domain
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = run_id
        if self.domain:
            self.run_dir = self.output_dir / self.domain / run_id
        else:
            self.run_dir = self.output_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "recommendations").mkdir(exist_ok=True)
    
    def save_recommendation_path(self, 
                                student_id: str,
                                path: List[Dict[str, Any]],
                                metadata: Optional[Dict[str, Any]] = None):
        output = {
            'student_id': student_id,
            'path': path,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        file_path = self.run_dir / "recommendations" / f"path_{student_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def save_generated_exercise(self,
                               exercise_id: str,
                               exercise: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None):
        (self.run_dir / "exercises").mkdir(exist_ok=True)
        output = {
            'exercise_id': exercise_id,
            'exercise': exercise,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        file_path = self.run_dir / "exercises" / f"exercise_{exercise_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def save_metrics(self,
                    metrics: Dict[str, Any],
                    filename: str = "metrics.json"):
        (self.run_dir / "metrics").mkdir(exist_ok=True)
        output = {
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id
        }
        
        file_path = self.run_dir / "metrics" / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def save_student_profile(self,
                           student_id: str,
                           profile: Dict[str, Any]):
        (self.run_dir / "profiles").mkdir(exist_ok=True)
        output = {
            'student_id': student_id,
            'profile': profile,
            'timestamp': datetime.now().isoformat()
        }
        
        file_path = self.run_dir / "profiles" / f"profile_{student_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    
    def get_run_dir(self) -> Path:
        return self.run_dir
