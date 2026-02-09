"""Student simulator: response simulation + learning gain (prereq coverage + response)."""
from typing import Callable, List, Optional

RESPONSE_FACTOR_WRONG = 0.6
PREREQ_COVERAGE_FLOOR = 0.4
BASE_PROB_FLOOR = 0.15
D_WRONG_BONUS_CAP = 3.5
D_WRONG_STEPS = 10
D_WRONG_INCREMENT = 0.25


class StudentSimulator:
    """Simulate response (KT + bonus vs 0.5) and compute learning gain."""
    
    def __init__(self, kt_model, kg_structure: dict, concept2id: dict, id2concept: dict, logger=None):
        self.kt_model = kt_model
        self.kg_structure = kg_structure or {}
        self.concept2id = concept2id or {}
        self.id2concept = id2concept or {}
        self.logger = logger
    
    def _get_prereq_coverage(self, learner, bundle: List[int], get_mastery: Callable[[int], float]) -> float:
        """Average prereq mastery for bundle concepts (floor PREREQ_COVERAGE_FLOOR)."""
        if not bundle:
            return max(0.5, PREREQ_COVERAGE_FLOOR)
        all_prereq_masteries = []
        for cid in bundle:
            prereq_strs = self.kg_structure.get(str(cid), [])
            if not prereq_strs:
                all_prereq_masteries.append(0.5)
                continue
            prereq_ids = []
            for s in prereq_strs:
                pid = self.concept2id.get(s)
                if pid is not None:
                    prereq_ids.append(pid)
                elif isinstance(s, str) and s.isdigit():
                    prereq_ids.append(int(s))
            if not prereq_ids:
                all_prereq_masteries.append(0.5)
                continue
            masteries = [get_mastery(pid) for pid in prereq_ids]
            all_prereq_masteries.extend(masteries)
        if not all_prereq_masteries:
            raw = 0.5
        else:
            raw = sum(all_prereq_masteries) / len(all_prereq_masteries)
        return max(raw, PREREQ_COVERAGE_FLOOR)
    
    def compute_learning_gain(
        self, learner, bundle: List[int], response: int, get_mastery: Callable[[int], float],
        base_gain: float = 0.06
    ) -> float:
        """Learning gain = base_gain * prereq_coverage * response_factor * wrong_count_bonus."""
        prereq_coverage = self._get_prereq_coverage(learner, bundle, get_mastery)
        response_factor = 1.0 if response == 1 else RESPONSE_FACTOR_WRONG
        gain = base_gain * prereq_coverage * response_factor
        if response == 0 and bundle:
            if not hasattr(learner, '_wrong_count'):
                learner._wrong_count = {}
            max_wrong = max(learner._wrong_count.get(cid, 0) for cid in bundle)
            wrong_bonus = 1.0 + min(max(max_wrong - 1, 0), D_WRONG_STEPS) * D_WRONG_INCREMENT
            wrong_bonus = min(max(1.0, wrong_bonus), D_WRONG_BONUS_CAP)
            gain *= wrong_bonus
        
        if self.logger:
            self.logger.debug(
                f"    Learning gain: prereq_coverage={prereq_coverage:.4f}, "
                f"response_factor={response_factor}, gain={gain:.4f}"
            )
        return gain
    
    def simulate_response(
        self,
        learner,
        bundle: List[int],
        target_concept: int,
        exercise_history: List,
        answer_history: List,
        get_mastery: Callable[[int], float],
    ) -> int:
        """Simulate response: KT + bonus vs 0.5. Update learner._learning_bonus. Returns 0 or 1."""
        bundle_concepts = bundle if bundle else [target_concept]
        
        try:
            if exercise_history and answer_history:
                kt_result = self.kt_model.forward(
                    exercise_inputs=exercise_history,
                    answers=answer_history,
                    concept_ids=bundle_concepts
                )
                concept_probs = [kt_result['mastery_probs'][i].item() for i in range(len(bundle_concepts))]
            else:
                concept_probs = [get_mastery(c) for c in bundle_concepts]
            
            if not hasattr(learner, '_learning_bonus'):
                learner._learning_bonus = {}
            
            adjusted_probs = []
            for i, cid in enumerate(bundle_concepts):
                base_prob = concept_probs[i]
                base_clamped = max(base_prob, BASE_PROB_FLOOR)
                bonus = learner._learning_bonus.get(cid, 0.0)
                adjusted_probs.append(min(1.0, base_clamped + bonus))
            
            avg_prob = sum(adjusted_probs) / len(adjusted_probs)
            response = 1 if avg_prob >= 0.5 else 0
            
            if response == 0:
                if not hasattr(learner, '_wrong_count'):
                    learner._wrong_count = {}
                for cid in bundle_concepts:
                    learner._wrong_count[cid] = learner._wrong_count.get(cid, 0) + 1
            
            gain = self.compute_learning_gain(learner, bundle_concepts, response, get_mastery)
            for cid in bundle_concepts:
                learner._learning_bonus[cid] = learner._learning_bonus.get(cid, 0.0) + gain
            
            if self.logger:
                self.logger.debug(
                    f"    KT prediction: bundle={bundle_concepts}, "
                    f"base_probs={[f'{p:.4f}' for p in concept_probs]}, "
                    f"avg_prob={avg_prob:.4f}, response={response}"
                )
        except Exception as e:
            if self.logger:
                self.logger.warning(f"KT prediction failed: {e}, using cached state")
            state = getattr(learner, '_state', {})
            concept_probs = [state.get(str(c), 0.5) for c in bundle_concepts]
            avg_prob = sum(concept_probs) / len(concept_probs)
            response = 1 if avg_prob >= 0.5 else 0
            if response == 0:
                if not hasattr(learner, '_wrong_count'):
                    learner._wrong_count = {}
                for cid in bundle_concepts:
                    learner._wrong_count[cid] = learner._wrong_count.get(cid, 0) + 1
            gain = self.compute_learning_gain(learner, bundle_concepts, response, get_mastery)
            for cid in bundle_concepts:
                if not hasattr(learner, '_learning_bonus'):
                    learner._learning_bonus = {}
                learner._learning_bonus[cid] = learner._learning_bonus.get(cid, 0.0) + gain
        
        return response
