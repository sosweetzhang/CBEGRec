"""
CBEGRec main pipeline: learning path recommendation via concept bundling and exercise generation.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.disable_proxy import disable_proxy, ensure_no_proxy
disable_proxy()

import json
import copy
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import random

from config import load_config, get_llm_config
from src.utils.logger import AppLogger
from src.utils.checkpoint import CheckpointManager
from src.utils.output_manager import OutputManager
from src.ecge.ecge import ECGE
from src.translator.ns_translator import NSTranslator
from src.gbp.gbp_planner import GBPPlanner
from src.aga.aga_generator import AGAGenerator
from src.data.data_loader import RecordsDataLoader
from src.simulator import StudentSimulator

class Pipeline:
    """CBEGRec pipeline: goal-oriented bundling and path-aware exercise generation."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path)
        self.logger = AppLogger("Main", level=self._get_log_level())
        self.checkpoint_manager = CheckpointManager(
            self.config['model']['checkpoint_dir']
        )
        domain = self.config.get('data', {}).get('domain', 'default')
        self.output_manager = OutputManager(
            self.config['logging']['output_dir'],
            domain=domain
        )
        data_config = self.config['data']
        domain = data_config['domain']
        base_path = Path(data_config['base_path']) / domain
        
        self.records_path = base_path / data_config['records_file']
        self.problem_info_path = base_path / data_config['problem_info_file']
        self.concept2id_path = base_path / data_config['concept2id_file']
        self.id2concept_path = base_path / data_config.get('id2concept_file', 'id2concept.json')
        self.kg_path = base_path / data_config.get('kg_structure_file', 'KG_structure.json')
        self.data_loader = RecordsDataLoader(
            str(self.records_path),
            str(self.problem_info_path),
            str(self.concept2id_path)
        )
        
        with open(self.problem_info_path, 'r', encoding='utf-8') as f:
            self.problem_info = json.load(f)
        self.concept2id = self.data_loader.concept2id
        self.id2concept = self.data_loader.id2concept
        if self.kg_path.exists():
            with open(self.kg_path, 'r', encoding='utf-8') as f:
                self.kg_structure = json.load(f)
            self.logger.info(f"Loaded KG from {self.kg_path}")
        else:
            self.logger.warning(f"KG_structure.json not found at {self.kg_path}, attempting to generate...")
            try:
                from src.kg.hybrid_kg_construction import HybridKGConstruction
                concept2id = self.data_loader.concept2id
                concepts_list = [f'"{name}": {cid}' for name, cid in concept2id.items()]
                kg_builder = HybridKGConstruction(self.config, self.logger)
                self.kg_structure = kg_builder.construct_hybrid_kg(
                    text_data=concepts_list,
                    records_path=str(self.records_path),
                    problem_info_path=str(self.problem_info_path),
                    concept2id_path=str(self.concept2id_path),
                    existing_concepts=concept2id
                )
                
                if not self.kg_structure:
                    raise RuntimeError("KG generation returned empty result")
                self.kg_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.kg_path, 'w', encoding='utf-8') as f:
                    json.dump(self.kg_structure, f, ensure_ascii=False, indent=2)
                self.logger.info(f"Generated and saved KG to {self.kg_path}")
            except Exception as e:
                self.logger.error(f"Failed to generate KG: {e}")
                raise RuntimeError(
                    f"Failed to generate KG: {e}. "
                    f"Please check LLM configuration and network connection."
                ) from e
        self._init_modules()
        
        self.logger.info("Pipeline initialized")
    
    def _get_log_level(self):
        """Get log level from config"""
        level_str = self.config['logging'].get('level', 'INFO')
        import logging
        return getattr(logging, level_str)
    
    def _parse_device(self, device_str: str) -> str:
        """Parse device string: auto, cpu, cuda, cuda:0, 0, etc. Returns normalized device string."""
        device_str = device_str.strip().lower()
        
        if device_str == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device_str == "cpu":
            return "cpu"
        elif device_str.startswith("cuda"):
            if torch.cuda.is_available():
                return device_str
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
        elif device_str.isdigit():
            gpu_id = int(device_str)
            if torch.cuda.is_available():
                if gpu_id < torch.cuda.device_count():
                    return f"cuda:{gpu_id}"
                else:
                    self.logger.warning(f"GPU {gpu_id} not found, using cuda:0")
                    return "cuda:0"
            else:
                self.logger.warning("CUDA not available, falling back to CPU")
                return "cpu"
        else:
            self.logger.warning(f"Unknown device '{device_str}', using auto detection")
            return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _init_modules(self):
        """Initialize pipeline modules."""
        num_concepts = len(self.concept2id)
        
        train_config = self.config.get('training', {})
        config_device = train_config.get('device', 'auto')
        device = self._parse_device(config_device)
        
        self.logger.info(f"Using device: {device}")
        if device.startswith("cuda"):
            gpu_name = torch.cuda.get_device_name(torch.device(device))
            self.logger.info(f"GPU: {gpu_name}")
        
        domain = self.config['data']['domain']
        base_model_dir = Path(self.config['model'].get('base_model_dir', './models'))
        self.ecge = ECGE(
            num_concepts=num_concepts,
            hidden_dim=self.config['ecge']['hidden_dim'],
            bert_path=self.config['model']['bert_path'],
            device=device
        )
        
        ecge_50_path = base_model_dir / domain / "ecge_50"
        ecge_50_file = ecge_50_path / "Trained_E_DKT_model.pt"
        
        if ecge_50_file.exists():
            try:
                self._load_model_weights(self.ecge, ecge_50_file)
                self.ecge.eval()
                self.logger.info(f"Loaded E-CGE model (Recommender, 50%) from {ecge_50_file}")
            except Exception as e:
                self.logger.warning(f"Could not load E-CGE model: {e}, using random initialization")
        else:
            self.logger.warning(f"E-CGE model not found at {ecge_50_file}, please train first!")
        self.kt_model = ECGE(
            num_concepts=num_concepts,
            hidden_dim=self.config['ecge']['hidden_dim'],
            bert_path=self.config['model']['bert_path'],
            device=device
        )
        
        kt_80_path = base_model_dir / domain / "ecge_kt_80"
        kt_80_file = kt_80_path / "Trained_E_DKT_model.pt"
        
        if kt_80_file.exists():
            try:
                self._load_model_weights(self.kt_model, kt_80_file)
                self.kt_model.eval()
                self.logger.info(f"Loaded KT model (Simulator, 80%) from {kt_80_file}")
            except Exception as e:
                self.logger.warning(f"Could not load KT model: {e}, using random initialization")
        else:
            self.logger.warning(f"KT model not found at {kt_80_file}, please train first!")
        self.translator = NSTranslator(
            threshold_low=self.config['ecge']['uncertainty_threshold_low'],
            threshold_high=self.config['ecge']['uncertainty_threshold_high']
        )
        self.gbp = GBPPlanner(
            self.kg_structure,
            self.config,
            self.logger
        )
        self.aga = AGAGenerator(
            self.problem_info,
            self.config,
            self.logger
        )
        
        self.simulator = StudentSimulator(
            kt_model=self.kt_model,
            kg_structure=self.kg_structure,
            concept2id=self.concept2id,
            id2concept=self.id2concept,
            logger=self.logger
        )
        
        self.learner_group = None
        self._Learner_class = None
        self.conceptids = self.concept2id
    
    def _load_model_weights(self, model, model_file):
        """Load model weights from checkpoint file."""
        checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    
    def _init_learner_class(self):
        """Lazy-init Learner class (avoids EduSim import at __init__)."""
        import sys
        import importlib.util
        
        learner_file = Path(__file__).parent.parent / "EduSim" / "Envs" / "KES_Mechanical_Physics" / "meta" / "Learner.py"
        
        for mod_name in ['EduSim', 'EduSim.Envs', 'EduSim.Envs.meta', 
                         'EduSim.Envs.KES_Mechanical_Physics', 
                         'EduSim.Envs.KES_Mechanical_Physics.meta']:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = type(sys)(mod_name)
        
        class MetaLearner:
            def __init__(self, user_id=None):
                self.id = user_id
        
        class MetaInfinityLearnerGroup:
            pass
        
        sys.modules['EduSim.Envs.meta'].MetaLearner = MetaLearner
        sys.modules['EduSim.Envs.meta'].MetaInfinityLearnerGroup = MetaInfinityLearnerGroup
        
        # Load Learner module
        spec = importlib.util.spec_from_file_location("Learner", learner_file)
        learner_module = importlib.util.module_from_spec(spec)
        learner_module.EduSim = sys.modules['EduSim']
        learner_module.np = np
        learner_module.copy = __import__('copy')
        learner_module.json = json
        learner_module.os = __import__('os')
        learner_module.random = __import__('random')
        spec.loader.exec_module(learner_module)
        
        self._Learner_class = learner_module.Learner
        self.logger.debug("Learner class initialized successfully")

    def get_mastery_probs(self, exercise_list: List[str], answer_list: List[int]) -> Dict[int, float]:
        """Compute concept mastery from E-CGE (recommender view)."""
        try:
            result = self.ecge.forward(exercise_list, answer_list, concept_ids=None)
            return {
                cid: result['mastery_probs'][i].item()
                for i, cid in enumerate(result['concept_ids'])
            }
        except Exception as e:
            self.logger.warning(f"get_mastery_probs failed: {e}, returning empty mastery")
            return {cid: 0.0 for cid in self.concept2id.values()}
    
    def get_kt_mastery_probs(self, exercise_list: List[str], answer_list: List[int]) -> Dict[int, float]:
        """Compute concept mastery from KT (80%) (simulator/ground-truth view)."""
        try:
            result = self.kt_model.forward(exercise_list, answer_list, concept_ids=None)
            return {
                cid: result['mastery_probs'][i].item()
                for i, cid in enumerate(result['concept_ids'])
            }
        except Exception as e:
            self.logger.warning(f"get_kt_mastery_probs failed: {e}, returning empty mastery")
            return {cid: 0.0 for cid in self.concept2id.values()}
    
    def select_target_concept(
        self,
        seq_len: int,
        question_ids: List[int],
        initial_exercises: List[str],
        initial_answers: List[int],
    ) -> Optional[Tuple[int, str]]:
        """Select one unmastered target concept in 80-100% range. Returns (target_concept_id, name) or None."""
        if not initial_exercises or not initial_answers:
            return None
        threshold = float(self.config.get('ecge', {}).get('mastery_threshold', 0.7))
        start_idx = int(0.8 * seq_len)
        if start_idx >= len(question_ids):
            return None
        concepts_in_range = set()
        for idx in range(start_idx, len(question_ids)):
            qid = question_ids[idx]
            if str(qid) not in self.problem_info:
                continue
            for cname in self.problem_info[str(qid)].get('concepts', []):
                cid = self.concept2id.get(cname)
                if cid is not None:
                    concepts_in_range.add(cid)
        if not concepts_in_range:
            target_qid = question_ids[start_idx]
            if str(target_qid) in self.problem_info:
                target_concepts = self.problem_info[str(target_qid)].get('concepts', [])
                if target_concepts:
                    cname = target_concepts[0]
                    cid = self.concept2id.get(cname)
                    if cid is not None and 0 <= cid < len(self.concept2id):
                        return (cid, cname)
            return None
        kt_mastery = self.get_kt_mastery_probs(initial_exercises, initial_answers)
        unmastered = [c for c in concepts_in_range if kt_mastery.get(c, 0.0) < threshold]
        
        if not unmastered:
            self.logger.debug(
                f"All concepts in 80-100% range are already mastered by KT (threshold={threshold}), "
                f"no target concept needed"
            )
            return None
        
        kg = self.kg_structure
        id2c = self.id2concept
        c2id = self.concept2id
        
        def prereqs_satisfied(cid: int) -> bool:
            cname = id2c.get(cid)
            if not cname or cname not in kg:
                return True
            for prereq_name in kg[cname]:
                prereq_id = c2id.get(prereq_name)
                if prereq_id is None:
                    continue
                if kt_mastery.get(prereq_id, 0.0) < threshold:
                    return False
            return True
        
        candidates = [c for c in unmastered if prereqs_satisfied(c)]
        if not candidates:
            candidates = unmastered
        num_concepts = len(self.concept2id)
        before_filter = len(candidates)
        candidates = [c for c in candidates if 0 <= c < num_concepts]
        if before_filter > len(candidates):
            self.logger.warning(
                f"Excluded {before_filter - len(candidates)} target(s) with concept_id >= {num_concepts} "
                f"(model cannot track mastery for these concepts)"
            )
        
        if not candidates:
            self.logger.debug("No valid candidates after KG filtering, cannot select target concept")
            return None
        
        cid = random.choice(candidates)
        cname = id2c.get(cid, f"Concept_{cid}")
        return (cid, cname)
    
    def recommend_path(self,
                      student_id: str,
                      initial_exercises: List[str],
                      initial_answers: List[int],
                      target_concept: int,
                      max_steps: Optional[int] = None,
                      initial_question_ids: List[int] = None) -> List[Dict]:
        """Recommend learning path (real student simulator). Returns list of step records."""
        if max_steps is None:
            max_steps = self.config['recommendation']['max_steps']
        
        target_concept_name = self.id2concept.get(target_concept, f"Concept_{target_concept}")
        self.logger.info(f"Starting recommendation for student {student_id}")
        self.logger.info(f"Target: {target_concept_name} (ID: {target_concept})")
        
        if self._Learner_class is None:
            self._init_learner_class()
        
        Learner = self._Learner_class
        
        if initial_question_ids is None:
            initial_question_ids = [0] * len(initial_exercises)
        
        initial_logs = [
            list(initial_question_ids),
            list(initial_exercises),
            list(initial_answers)
        ]
        
        target_question_id = None
        for qid, qinfo in self.problem_info.items():
            if target_concept in [self.concept2id.get(c, -1) for c in qinfo.get('concepts', [])]:
                target_question_id = int(qid)
                break
        
        if target_question_id is None:
            target_question_id = max([int(k) for k in self.problem_info.keys()]) + 1
        
        learner = Learner(
            initial_log=copy.deepcopy(initial_logs),
            learning_target={target_question_id},
            _id=student_id,
            seed=None
        )
        
        exercise_history = initial_exercises.copy()
        answer_history = initial_answers.copy()
        
        self._update_learner_state(learner, target_concept, exercise_history, answer_history)
        initial_target_mastery_kt = self._get_concept_mastery_from_learner(learner, target_concept)
        try:
            init_result = self.ecge.forward(exercise_history, answer_history, concept_ids=None)
            init_mastery_probs = {
                cid: init_result['mastery_probs'][i].item()
                for i, cid in enumerate(init_result['concept_ids'])
            }
            initial_target_mastery = float(init_mastery_probs.get(target_concept, initial_target_mastery_kt))
            self.logger.debug(
                f"Initial E-CGE calculation: history_len={len(exercise_history)}, "
                f"target_concept={target_concept}, mastery={initial_target_mastery:.6f}"
            )
        except Exception:
            initial_target_mastery = initial_target_mastery_kt
        
        target_concept_name = self.id2concept.get(target_concept, f"Concept_{target_concept}")
        self.logger.info(f"Initial mastery: KT={initial_target_mastery_kt:.4f}, E-CGE={initial_target_mastery:.4f}")
        
        mastery_threshold = float(self.config['ecge']['mastery_threshold'])
        if initial_target_mastery >= mastery_threshold:
            self.logger.info(
                f"Target concept already mastered (E-CGE estimate): "
                f"E-CGE={initial_target_mastery:.4f}, KT={initial_target_mastery_kt:.4f} "
                f"(threshold={mastery_threshold:.4f}). Skipping recommendation."
            )
            return [{'skipped': True, 'reason': 'already_mastered', 
                    'initial_kt_mastery': initial_target_mastery_kt,
                    'initial_ecge_mastery': initial_target_mastery}]
        
        recommendation_path = []
        
        previous_target_mastery_ecge = initial_target_mastery
        
        for step in range(max_steps):
            self.logger.info("")
            self.logger.info(f"  Step {step + 1}/{max_steps}")
            try:
                result = self.ecge.forward(
                    exercise_history,
                    answer_history,
                    concept_ids=None
                )
                
                mastery_probs = {
                    cid: result['mastery_probs'][i].item()
                    for i, cid in enumerate(result['concept_ids'])
                }
                uncertainties = {
                    cid: result['uncertainties'][i].item()
                    for i, cid in enumerate(result['concept_ids'])
                }
                
                target_mastery_current = mastery_probs.get(target_concept, 0.0)
                
                mastery_values = list(mastery_probs.values())
                mastery_min = min(mastery_values) if mastery_values else 0
                mastery_max = max(mastery_values) if mastery_values else 0
                mastery_avg = sum(mastery_values) / len(mastery_values) if mastery_values else 0
                
                self.logger.info(
                    f"    History length={len(exercise_history)}, "
                    f"Target mastery={target_mastery_current:.4f}"
                )
                self.logger.info(
                    f"    E-CGE mastery distribution: min={mastery_min:.4f}, max={mastery_max:.4f}, "
                    f"avg={mastery_avg:.4f}, total_concepts={len(mastery_probs)}"
                )
                if step == 0:
                    diff = abs(target_mastery_current - initial_target_mastery)
                    self.logger.debug(
                        f"Step 1 verification: Initial E-CGE={initial_target_mastery:.6f}, "
                        f"Step 1 E-CGE={target_mastery_current:.6f}, diff={diff:.6f}"
                    )
                    if diff > 1e-5:
                        self.logger.warning(
                            f"Step 1 E-CGE differs from Initial E-CGE by {diff:.6f} "
                            f"(expected to be identical since history hasn't changed)"
                        )
            except Exception as e:
                self.logger.error(f"E-CGE forward failed at step {step+1}: {e}")
                if step > 0 and recommendation_path:
                    last_step = recommendation_path[-1]
                    mastery_probs = last_step.get('mastery_probs', {})
                    uncertainties = last_step.get('uncertainties', {})
                else:
                    mastery_probs = {cid: 0.0 for cid in self.concept2id.values()}
                    uncertainties = {cid: 1.0 for cid in self.concept2id.values()}
            
            target_mastery_ecge = mastery_probs.get(target_concept, 0.0)
            target_mastery_kt = self._get_concept_mastery_from_learner(learner, target_concept)
            if target_mastery_ecge >= self.config['ecge']['mastery_threshold']:
                self.logger.info(f"    Target concept mastered (E-CGE estimate)! "
                              f"E-CGE={target_mastery_ecge:.4f}, KT={target_mastery_kt:.4f}")
                break
            
            self.logger.debug(
                f"    Target mastery E-CGE={target_mastery_ecge:.4f}, KT={target_mastery_kt:.4f}"
            )
            recent_bundles = [s['bundle'] for s in recommendation_path]
            bundle, bundle_selection_reason = self.gbp.plan_bundle(
                mastery_probs,
                uncertainties,
                target_concept,
                self.id2concept,
                recent_bundles=recent_bundles
            )
            
            if not bundle:
                self.logger.warning("    No bundle found, stopping")
                break
            
            bundle_concept_names = [self.id2concept.get(cid, str(cid)) for cid in bundle]
            self.logger.info(f"    Selected bundle: {bundle_concept_names}")
            self.logger.info(f"    Bundle selection reason: {bundle_selection_reason[:80]}..." if len(bundle_selection_reason) > 80 else f"    Bundle selection reason: {bundle_selection_reason}")
            
            bundle_mastery = {cid: mastery_probs.get(cid, 0.0) for cid in bundle}
            exercise, answer, exercise_explanation, is_valid, retrieval_success, reference_exercise_preview = self.aga.generate_exercise(
                bundle,
                self.id2concept,
                bundle_mastery,
                bundle_selection_reason=bundle_selection_reason
            )
            
            if not exercise:
                self.logger.warning("    Exercise generation failed, stopping")
                break
            
            retrieval_label = "success" if retrieval_success else "fail"
            self.logger.info(f"    Retrieval: {retrieval_label}")
            if retrieval_success and reference_exercise_preview:
                self.logger.info(f"    Reference exercise: {reference_exercise_preview}")
            
            self.logger.info(f"    Generated exercise:")
            exercise_preview = exercise[:200] + "..." if len(exercise) > 200 else exercise
            self.logger.info(f"      {exercise_preview}")
            if exercise_explanation:
                expl_preview = exercise_explanation[:100] + "..." if len(exercise_explanation) > 100 else exercise_explanation
                self.logger.info(f"    Exercise explanation: {expl_preview}")
            
            if self.problem_info:
                try:
                    max_id = max([int(k) for k in self.problem_info.keys() if k.isdigit()])
                    new_problem_id = max_id + 1
                except (ValueError, TypeError):
                    new_problem_id = 10000
            else:
                new_problem_id = 10000
            
            self.problem_info[str(new_problem_id)] = {
                'content': exercise,
                'concepts': [self.id2concept.get(cid, f"Concept_{cid}") for cid in bundle],
                'difficulty': 2
            }
            def get_mastery(cid):
                return self._get_concept_mastery_from_learner(learner, cid)
            student_response = self.simulator.simulate_response(
                learner=learner,
                bundle=bundle,
                target_concept=target_concept,
                exercise_history=exercise_history,
                answer_history=answer_history,
                get_mastery=get_mastery,
            )
            
            response_text = "correct" if student_response == 1 else "wrong"
            self.logger.info(f"    Student response: {response_text} ({student_response})")
            logs = learner.profile['logs']
            logs[0].append(int(new_problem_id))
            logs[1].append(str(exercise) if exercise is not None else "")
            logs[2].append(int(student_response))
            
            target_mastery_kt_before = self._get_concept_mastery_from_learner(learner, target_concept)
            
            exercise_history.append(exercise)
            answer_history.append(student_response)
            
            self._update_learner_state(learner, target_concept, exercise_history, answer_history)
            target_mastery_kt_updated = self._get_concept_mastery_from_learner(learner, target_concept)
            
            self.logger.info(f"    KT mastery change: {target_mastery_kt_before:.4f} -> {target_mastery_kt_updated:.4f} "
                           f"(change: {target_mastery_kt_updated - target_mastery_kt_before:+.4f})")
            
            try:
                updated_result = self.ecge.forward(exercise_history, answer_history, concept_ids=None)
                updated_mastery_probs = {
                    cid: updated_result['mastery_probs'][i].item()
                    for i, cid in enumerate(updated_result['concept_ids'])
                }
                current_target_mastery_ecge = updated_mastery_probs.get(target_concept, 0.0)
            except Exception:
                current_target_mastery_ecge = target_mastery_current
            step_record = {
                'step': step + 1,
                'bundle': bundle,
                'bundle_names': [self.id2concept.get(cid, f"Concept_{cid}") for cid in bundle],
                'bundle_selection_reason': bundle_selection_reason,
                'exercise': exercise,
                'answer': answer,
                'exercise_explanation': exercise_explanation,
                'student_response': student_response,
                'retrieval_success': retrieval_success,
                'reference_exercise': reference_exercise_preview,
                'mastery_probs': {k: float(v) for k, v in mastery_probs.items()},
                'uncertainties': {k: float(v) for k, v in uncertainties.items()},
                'target_mastery': float(current_target_mastery_ecge),
                'target_mastery_kt': float(target_mastery_kt_updated),
                'target_mastery_kt_before': float(target_mastery_kt_before),
                'problem_id': new_problem_id
            }
            recommendation_path.append(step_record)
            
            mastery_change_ecge = current_target_mastery_ecge - previous_target_mastery_ecge
            mastery_change_kt = target_mastery_kt_updated - target_mastery_kt
            
            self.logger.info(f"    Mastery updated: {previous_target_mastery_ecge:.4f} -> {current_target_mastery_ecge:.4f} "
                          f"(change: {mastery_change_ecge:+.4f})")
            
            self.logger.info(f"    Step {step + 1} completed: Bundle={bundle_concept_names}, "
                          f"Response={response_text}, Mastery={current_target_mastery_ecge:.4f}")
            previous_target_mastery_ecge = current_target_mastery_ecge
            if (step + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.config['recommendation']['enable_checkpoint']:
                if (step + 1) % self.config['recommendation']['checkpoint_interval'] == 0:
                    state = {
                        'exercise_history': exercise_history,
                        'answer_history': answer_history,
                        'step': step + 1
                    }
                    self.checkpoint_manager.save_inference_state(state, step + 1)
        
        final_target_mastery_kt = self._get_concept_mastery_from_learner(learner, target_concept)
        
        try:
            final_result = self.ecge.forward(exercise_history, answer_history, concept_ids=None)
            final_mastery_probs = {
                cid: final_result['mastery_probs'][i].item()
                for i, cid in enumerate(final_result['concept_ids'])
            }
            final_target_mastery = float(final_mastery_probs.get(target_concept, final_target_mastery_kt))
        except Exception:
            final_target_mastery = float(
                recommendation_path[-1].get('target_mastery_kt', 
                recommendation_path[-1].get('target_mastery', 0.0))
            ) if recommendation_path else final_target_mastery_kt

        goal_completed = final_target_mastery_kt >= self.config['ecge']['mastery_threshold']
        
        self.logger.info("")
        self.logger.info(f"Recommendation completed: {len(recommendation_path)} steps")
        self.logger.info(f"Final mastery: KT={final_target_mastery_kt:.4f}, E-CGE={final_target_mastery:.4f}, "
                        f"Goal completed: {goal_completed}")

        self.output_manager.save_recommendation_path(
            student_id,
            recommendation_path,
            metadata={
                'target_concept': target_concept,
                'target_concept_name': target_concept_name,
                'total_steps': len(recommendation_path),
                'goal_completed': goal_completed,
                'initial_target_mastery': float(initial_target_mastery),
                'initial_target_mastery_kt': float(initial_target_mastery_kt),
                'final_target_mastery': float(final_target_mastery),
                'final_target_mastery_kt': float(final_target_mastery_kt),
                'mastery_threshold': float(self.config['ecge']['mastery_threshold'])
            }
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return recommendation_path
    
    def _update_learner_state(self, learner, target_concept=None, exercise_history=None, answer_history=None):
        """Update learner state using KT model (80% data)."""
        if exercise_history is not None and answer_history is not None:
            ques_text = list(exercise_history)
            ans = list(answer_history)
        else:
            logs = learner.profile['logs']
            ans = list(logs[2])
            ques_text_raw = logs[1]
            ques_text = []
            for item in ques_text_raw:
                if isinstance(item, str) and item.strip():
                    ques_text.append(item.strip())
                elif isinstance(item, int):
                    qid_str = str(item)
                    if qid_str in self.problem_info:
                        ques_text.append(self.problem_info[qid_str].get('content', '').strip())
                    else:
                        ques_text.append('')
                else:
                    text = str(item) if item is not None else ''
                    ques_text.append(text.strip())
        
        if not ques_text:
            self.logger.warning("No question texts found, cannot update state")
            return
        min_len = min(len(ques_text), len(ans))
        ques_text = ques_text[:min_len]
        ans = ans[:min_len]
        try:
            result = self.kt_model.forward(
                exercise_inputs=ques_text,
                answers=ans,
                concept_ids=None
            )
            new_state = {}
            mastery_probs = result['mastery_probs']
            concept_ids = result['concept_ids']
            for i, cid in enumerate(concept_ids):
                new_state[str(cid)] = mastery_probs[i].item()
            learner._state = new_state
            self.logger.debug(f"KT state updated with {len(ques_text)} exercises using kt_model (80%)")
            if target_concept is not None and str(target_concept) in new_state:
                self.logger.debug(f"  Target concept {target_concept} mastery (KT): {new_state[str(target_concept)]:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to update learner state: {e}")
            import traceback
            self.logger.error(f"Error traceback: {traceback.format_exc()}")
            raise RuntimeError(
                f"Failed to update learner state: {e}. "
                f"This is a critical error."
            ) from e
    
    def _get_concept_mastery_from_learner(self, learner, concept_id: int) -> float:
        """Get concept mastery from learner state. State format: {str(concept_id): mastery_prob}."""
        concept_name = self.id2concept.get(concept_id, f"Concept_{concept_id}")
        state = getattr(learner, '_state', None)
        if state is None:
            state = getattr(learner, 'state', None)
        if state is None and hasattr(learner, 'profile') and isinstance(learner.profile, dict):
            state = learner.profile.get('state', None)
        
        if isinstance(state, dict) and state:
            mastery = state.get(str(concept_id))
            if mastery is None:
                mastery = state.get(concept_name)
            if mastery is None:
                mastery = state.get(concept_id)
            if mastery is not None:
                self.logger.debug(f"Found mastery for concept {concept_id} ({concept_name}): {mastery:.4f}")
                return float(mastery)
        elif isinstance(state, list) and state:
            if 0 <= concept_id < len(state):
                mastery_val = float(state[concept_id])
                self.logger.debug(f"Found mastery for concept {concept_id} (list index): {mastery_val:.4f}")
                return mastery_val
        available_keys = list(state.keys())[:10] if isinstance(state, dict) else f"list length {len(state)}" if isinstance(state, list) else "N/A"
        self.logger.warning(f"No mastery found for concept {concept_id} ({concept_name}) in learner state")
        self.logger.warning(f"  State type: {type(state)}, available keys/info: {available_keys}")
        return 0.0