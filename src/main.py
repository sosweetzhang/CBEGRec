from __future__ import annotations

import copy
import ast
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config, get_llm_config
from prompts import build_messages
from src.concept_bundling.goal_oriented_concept_bundling import GoalOrientedConceptBundling
from src.data.data_loader import Learner, RecordsDataLoader
from src.exercise_generation.path_aware_exercise_generation import PathAwareExerciseGeneration
from src.kg.hybrid_kg_construction import HybridKGConstruction
from src.simulator.student_simulator import StudentSimulator
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import AppLogger
from src.utils.llm_client import OpenAILLMClient
from src.utils.output_manager import OutputManager


def _deep_update(base: dict, updates: dict) -> dict:
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class DeterministicLLMClient:

    def complete(self, role: str, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        prompt = messages[-1]["content"] if messages else ""
        if role == "BundleSelection":
            return "Bundle 1; Reason: deterministic fallback for offline execution."
        if role == "Teacher":
            bundle_line = next((line for line in prompt.splitlines() if line.startswith("Target concept bundle:")), "")
            bundle_names = bundle_line.split(":", 1)[-1].strip() if ":" in bundle_line else "the target bundle"
            return (
                f"Exercise: Create a practice item for {bundle_names}.\n"
                f"Answer: 1\n"
                f"Explanation: This exercise reinforces the selected concept bundle and matches the current learning step."
            )
        if role == "Solver":
            return "1"
        if role == "Critic":
            return "Valid: True\nReason: deterministic offline validation."
        if role == "KGRefinement":
            return "Added edges: []\nRemoved edges: []\nNotes: offline fallback."
        return "Valid: True\nReason: offline fallback."


class _Scalar(float):
    def item(self) -> float:
        return float(self)


class _SimpleECGE:

    def __init__(self, num_concepts: int):
        self.num_concepts = num_concepts

    def eval(self):
        return self

    def load_state_dict(self, *args, **kwargs):
        return None

    def forward(self, exercise_inputs, answers, concept_ids=None, **kwargs):
        if concept_ids is None:
            concept_ids_all = list(range(self.num_concepts))
        else:
            concept_ids_all = [int(cid.item() if hasattr(cid, "item") else cid) for cid in concept_ids]
        answers = list(answers or [])
        correctness = sum(1 for answer in answers if int(answer) == 1) / len(answers) if answers else 0.0
        progress = min(1.0, len(answers) / 20.0)
        mastery_probs = []
        uncertainties = []
        for cid in concept_ids_all:
            concept_offset = ((int(cid) * 37) % 11) / 100.0
            prob = max(0.05, min(0.95, 0.22 + 0.33 * correctness + 0.18 * progress + concept_offset))
            mastery_probs.append(_Scalar(prob))
            if 0.0 < prob < 1.0:
                uncertainty = -(prob * math.log2(prob) + (1.0 - prob) * math.log2(1.0 - prob))
            else:
                uncertainty = 0.0
            uncertainties.append(_Scalar(uncertainty))
        return {
            "hidden_state": None,
            "mastery_probs": mastery_probs,
            "uncertainties": uncertainties,
            "concept_ids": concept_ids_all,
        }


class Pipeline:

    def __init__(self, config_path: Optional[str] = None, config_overrides: Optional[dict] = None):
        self.config = load_config(config_path)
        if config_overrides:
            _deep_update(self.config, config_overrides)
        self.logger = AppLogger("CBEGRec", level=self._get_log_level())
        self.checkpoint_manager = CheckpointManager(self.config["model"]["checkpoint_dir"])
        domain = self.config.get("data", {}).get("domain", "default")
        self.output_manager = OutputManager(self.config["logging"]["output_dir"], domain=domain)

        data_config = self.config["data"]
        base_path = Path(data_config["base_path"]) / data_config["domain"]
        self.records_path = base_path / data_config["records_file"]
        self.problem_info_path = base_path / data_config["problem_info_file"]
        self.problem_bank_path = base_path / data_config.get("problem_file", "problem.json")
        self.problem2id_path = base_path / data_config.get("problem2id_file", "problem2id.json")
        self.concept2id_path = base_path / data_config["concept2id_file"]
        self.id2concept_path = base_path / data_config.get("id2concept_file", "id2concept.json")
        self.kg_path = base_path / data_config.get("kg_structure_file", "KG_structure.json")

        self.data_loader = RecordsDataLoader(str(self.records_path), str(self.problem_info_path), str(self.concept2id_path))
        with open(self.problem_info_path, "r", encoding="utf-8") as f:
            self.problem_info = json.load(f)
        self._merge_problem_bank_metadata(self.problem_bank_path, self.problem2id_path)
        self.concept2id = self.data_loader.concept2id
        self.id2concept = self.data_loader.id2concept
        self.seed = int(self.config.get("training", {}).get("seed", 2024))

        self.llm_client = self._build_llm_client()
        self.kg_structure = self._load_or_build_kg()
        self._init_models()
        self.concept_bundling = GoalOrientedConceptBundling(self.kg_structure, self.config, logger=self.logger, llm_client=self.llm_client)
        self.exercise_generation = PathAwareExerciseGeneration(self.problem_info, self.config, llm_client=self.llm_client, logger=self.logger)
        self.simulator = StudentSimulator(
            kt_model=self.kt_model,
            kg_structure=self.kg_structure,
            concept2id=self.concept2id,
            id2concept=self.id2concept,
            logger=self.logger,
        )
        self.logger.info("Pipeline initialized")

    def _get_log_level(self):
        level_str = self.config["logging"].get("level", "INFO")
        import logging

        return getattr(logging, level_str)

    def _build_llm_client(self):
        llm_cfg = self.config.get("llm", {})
        if llm_cfg.get("mock", False):
            return DeterministicLLMClient()
        env_cfg = get_llm_config()
        api_key = llm_cfg.get("api_key") or env_cfg.get("api_key")
        if not api_key:
            return DeterministicLLMClient()
        from openai import OpenAI

        model = llm_cfg.get("model") or env_cfg.get("model")
        base_url = llm_cfg.get("base_url") or env_cfg.get("base_url")
        client = OpenAI(api_key=api_key, base_url=base_url or None)
        return OpenAILLMClient(client, model)

    def _load_or_build_kg(self) -> Dict[str, List[str]]:
        rebuild = bool(self.config.get("kg_construction", {}).get("rebuild_on_start", False))
        if self.kg_path.exists() and not rebuild:
            with open(self.kg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        builder = HybridKGConstruction(self.config, self.logger, llm_client=self.llm_client)
        kg = builder.construct_hybrid_kg(
            text_data=[f'"{name}": {cid}' for name, cid in self.concept2id.items()],
            records_path=str(self.records_path),
            problem_info_path=str(self.problem_info_path),
            concept2id_path=str(self.concept2id_path),
            existing_concepts=self.concept2id,
        )
        self.kg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.kg_path, "w", encoding="utf-8") as f:
            json.dump(kg, f, ensure_ascii=False, indent=2)
        return kg

    def _load_model_weights(self, model, model_file):
        torch_module = getattr(self, "_torch", None)
        if torch_module is None:
            return
        checkpoint = torch_module.load(model_file, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)

    def _init_models(self):
        num_concepts = len(self.concept2id)
        try:
            import torch as torch_module  
            from src.ecge.ecge import ECGE as ECGEModel

            self._torch = torch_module
        except Exception as exc:
            self._torch = None
            self.logger.warning(f"Falling back to simple E-CGE interface for smoke tests: {exc}")
            self.ecge = _SimpleECGE(num_concepts)
            self.kt_model = _SimpleECGE(num_concepts)
            return
        device = self._parse_device(self.config.get("training", {}).get("device", "auto"))
        self.logger.info(f"Using device: {device}")
        try:
            self.ecge = ECGEModel(
                num_concepts=num_concepts,
                hidden_dim=self.config["ecge"]["hidden_dim"],
                bert_path=self.config["model"]["bert_path"],
                device=device,
            )
            self.kt_model = ECGEModel(
                num_concepts=num_concepts,
                hidden_dim=self.config["ecge"]["hidden_dim"],
                bert_path=self.config["model"]["bert_path"],
                device=device,
            )
        except Exception as exc:
            self._torch = None
            self.logger.warning(f"Falling back to simple E-CGE interface for smoke tests: {exc}")
            self.ecge = _SimpleECGE(num_concepts)
            self.kt_model = _SimpleECGE(num_concepts)
            return
        base_model_dir = Path(self.config["model"].get("base_model_dir", "./models"))
        domain = self.config["data"]["domain"]
        ecge_50_file = base_model_dir / domain / "ecge_50" / "Trained_E_DKT_model.pt"
        kt_80_file = base_model_dir / domain / "ecge_kt_80" / "Trained_E_DKT_model.pt"
        if ecge_50_file.exists():
            self._load_model_weights(self.ecge, ecge_50_file)
            self.ecge.eval()
        if kt_80_file.exists():
            self._load_model_weights(self.kt_model, kt_80_file)
            self.kt_model.eval()

    def _parse_device(self, device_str: str) -> str:
        torch_module = getattr(self, "_torch", None)
        if torch_module is None:
            return "cpu"
        device_str = (device_str or "auto").strip().lower()
        if device_str == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if device_str == "cpu":
            return "cpu"
        if device_str.startswith("cuda") and torch_module.cuda.is_available():
            return device_str
        if device_str.isdigit() and torch_module.cuda.is_available():
            gpu_id = int(device_str)
            return f"cuda:{gpu_id}" if gpu_id < torch_module.cuda.device_count() else "cuda:0"
        return "cpu"

    def get_mastery_probs(self, exercise_list: List[str], answer_list: List[int]) -> Dict[int, float]:
        result = self.ecge.forward(exercise_list, answer_list, concept_ids=None)
        return {cid: result["mastery_probs"][i].item() for i, cid in enumerate(result["concept_ids"])}

    def get_kt_mastery_probs(self, exercise_list: List[str], answer_list: List[int]) -> Dict[int, float]:
        result = self.kt_model.forward(exercise_list, answer_list, concept_ids=None)
        return {cid: result["mastery_probs"][i].item() for i, cid in enumerate(result["concept_ids"])}

    def select_target_concept(
        self,
        seq_len: int,
        question_ids: List[int],
        initial_exercises: List[str],
        initial_answers: List[int],
    ) -> Optional[Tuple[int, str]]:
        if not initial_exercises or not initial_answers:
            return None
        threshold = float(self.config.get("ecge", {}).get("mastery_threshold", 0.7))
        start_idx = int(0.8 * seq_len)
        if start_idx >= len(question_ids):
            return None
        concepts_in_range = set()
        for idx in range(start_idx, len(question_ids)):
            qid = question_ids[idx]
            info = self.problem_info.get(str(qid), {})
            for cname in info.get("concepts", []):
                cid = self.concept2id.get(cname)
                if cid is not None:
                    concepts_in_range.add(cid)
        if not concepts_in_range:
            return None
        mastery = self.get_kt_mastery_probs(initial_exercises, initial_answers)
        candidates = [cid for cid in concepts_in_range if mastery.get(cid, 0.0) < threshold]
        if not candidates:
            return None
        cid = random.choice(candidates)
        return cid, self.id2concept.get(cid, f"Concept_{cid}")

    def _get_concept_mastery_from_learner(self, learner, concept_id: int) -> float:
        state = getattr(learner, "_state", {}) or {}
        mastery = state.get(str(concept_id))
        if mastery is None:
            return 0.0
        return float(mastery)

    def _update_learner_state(self, learner, exercise_history: List[str], answer_history: List[int]):
        result = self.kt_model.forward(exercise_history, answer_history, concept_ids=None)
        learner._state = {str(cid): result["mastery_probs"][i].item() for i, cid in enumerate(result["concept_ids"])}

    def _merge_problem_bank_metadata(self, problem_bank_path: Path, problem2id_path: Path):
        if not problem_bank_path.exists() or not problem2id_path.exists():
            return
        with open(problem2id_path, "r", encoding="utf-8") as f:
            problem2id = {str(key): str(value) for key, value in json.load(f).items()}
        with open(problem_bank_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                local_id = problem2id.get(str(record.get("problem_id")))
                if local_id is None or local_id not in self.problem_info:
                    continue
                detail = record.get("detail") or {}
                if isinstance(detail, str):
                    try:
                        detail = ast.literal_eval(detail)
                    except (ValueError, SyntaxError):
                        detail = {}
                if not isinstance(detail, dict):
                    detail = {}
                answer = detail.get("answer") or record.get("answer")
                options = detail.get("option") or detail.get("options") or record.get("option") or record.get("options")
                if answer is not None:
                    self.problem_info[local_id]["answer"] = answer
                if isinstance(options, dict) and options:
                    self.problem_info[local_id]["options"] = options

    def _format_problem_content(self, problem: Dict) -> str:
        content = problem.get("content", "")
        options = problem.get("options")
        if isinstance(options, dict) and options:
            option_lines = [f"{key}. {value}" for key, value in options.items()]
            content = "\n".join([content, *option_lines])
        return content

    def _retrieve_generation_exercise(self, bundle: List[int]) -> Tuple[str, str]:
        bundle_names = {self.id2concept.get(cid, f"Concept_{cid}") for cid in bundle}
        best = None
        best_overlap = -1
        for problem in self.problem_info.values():
            concepts = set(problem.get("concepts", []))
            overlap = len(bundle_names & concepts)
            if overlap > best_overlap:
                best = problem
                best_overlap = overlap
        if not best or best_overlap <= 0:
            return "", ""
        return self._format_problem_content(best), best.get("answer", "")

    def recommend_path(
        self,
        student_id: str,
        initial_exercises: List[str],
        initial_answers: List[int],
        target_concept: int,
        max_steps: Optional[int] = None,
        initial_question_ids: List[int] = None,
        variant: str = "full",
    ) -> List[Dict]:
        max_steps = max_steps or self.config["recommendation"]["max_steps"]
        target_name = self.id2concept.get(target_concept, f"Concept_{target_concept}")
        if initial_question_ids is None:
            initial_question_ids = [0] * len(initial_exercises)
        initial_logs = [list(initial_question_ids), list(initial_exercises), list(initial_answers)]
        learner = Learner(initial_log=copy.deepcopy(initial_logs), learning_target={target_concept}, _id=student_id, seed=self.seed)
        exercise_history = list(initial_exercises)
        answer_history = list(initial_answers)
        self._update_learner_state(learner, exercise_history, answer_history)
        recommendation_path = []
        mastery_threshold = float(self.config["ecge"]["mastery_threshold"])

        for step in range(max_steps):
            result = self.ecge.forward(exercise_history, answer_history, concept_ids=None)
            mastery_probs = {cid: result["mastery_probs"][i].item() for i, cid in enumerate(result["concept_ids"])}
            target_mastery = mastery_probs.get(target_concept, 0.0)
            if target_mastery >= mastery_threshold:
                break

            recent_bundles = [step_record["b_t"] for step_record in recommendation_path]
            if variant in {"wo_cb", "wo_cbeg"}:
                b_t, j_t = self.concept_bundling.select_single_concept(mastery_probs, target_concept)
            else:
                b_t, j_t, _ = self.concept_bundling.plan_bundle(mastery_probs, target_concept, self.id2concept, recent_bundles=recent_bundles)
            if not b_t:
                break

            if variant in {"wo_eg", "wo_cbeg"}:
                e_t, a_ref_t = self._retrieve_generation_exercise(b_t)
                J_t = f"Retrieved exercise aligned with {', '.join(self.id2concept.get(cid, str(cid)) for cid in b_t)}."
                a_sol_t = a_ref_t
                valid_t = True
                retrieval_used = True
                reference_preview = e_t[:200] if e_t else None
            else:
                generated = self.exercise_generation.generate_exercise(b_t, self.id2concept, {cid: mastery_probs.get(cid, 0.0) for cid in b_t}, j_t)
                e_t = generated["e_t"]
                a_ref_t = generated["a_ref_t"]
                J_t = generated["J_t"]
                a_sol_t = generated["a_sol_t"]
                valid_t = generated["valid_t"]
                retrieval_used = generated["retrieval_used"]
                reference_preview = generated["reference_preview"]

            if not e_t:
                break

            def get_mastery(cid):
                return self._get_concept_mastery_from_learner(learner, cid)

            bundle_for_simulation = b_t if b_t else [target_concept]
            target_mastery_before = self._get_concept_mastery_from_learner(learner, target_concept)
            student_response = self.simulator.simulate_response(
                learner=learner,
                bundle=bundle_for_simulation,
                target_concept=target_concept,
                exercise_history=exercise_history,
                answer_history=answer_history,
                get_mastery=get_mastery,
            )
            exercise_history.append(e_t)
            answer_history.append(student_response)
            self._update_learner_state(learner, exercise_history, answer_history)
            current_target_mastery = self._get_concept_mastery_from_learner(learner, target_concept)
            step_record = {
                "step": step + 1,
                "b_t": b_t,
                "bundle_names": [self.id2concept.get(cid, f"Concept_{cid}") for cid in b_t],
                "j_t": j_t,
                "e_t": e_t,
                "a_ref_t": a_ref_t,
                "a_sol_t": a_sol_t,
                "J_t": J_t,
                "valid_t": valid_t,
                "student_response": student_response,
                "retrieval_used": retrieval_used,
                "reference_preview": reference_preview,
                "target_mastery_kt_before": float(target_mastery_before),
                "target_mastery_kt": float(current_target_mastery),
                "target_mastery": float(current_target_mastery),
            }
            recommendation_path.append(step_record)
        final_mastery = self._get_concept_mastery_from_learner(learner, target_concept)
        goal_completed = final_mastery >= mastery_threshold
        self.output_manager.save_recommendation_path(
            student_id,
            recommendation_path,
            metadata={
                "target_concept": target_concept,
                "target_concept_name": target_name,
                "total_steps": len(recommendation_path),
                "goal_completed": goal_completed,
                "final_target_mastery": float(final_mastery),
                "mastery_threshold": mastery_threshold,
                "variant": variant,
            },
        )
        return recommendation_path
