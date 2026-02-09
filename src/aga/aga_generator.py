"""
Auditor-Guided Generative Agent (AGA): Teacher-Solver-Critic loop.
RAG: retrieve reference exercises; fallback to free generation if retrieval fails.
"""
import json
from typing import Dict, List, Tuple, Optional

REFERENCE_EXERCISE_TRUNCATE_LEN = 200
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_llm_config
from prompts import build_messages, get_prompt, get_prompt_lang
from src.utils.logger import AppLogger

class AGAGenerator:
    """Auditor-guided generative agent."""
    
    def __init__(self,
                 problem_info: Dict,
                 config: dict,
                 logger: AppLogger = None):
        self.problem_info = problem_info
        self.config = config
        self.logger = logger or AppLogger("AGA")
        
        aga_config = config.get('aga', {})
        self.max_retries = aga_config.get('max_retries', 3)
        self.rag_top_k = aga_config.get('rag_top_k', 5)
        self.solver_temperature = aga_config.get('solver_temperature', 0.0)
        from src.utils.disable_proxy import ensure_no_proxy
        ensure_no_proxy()
        
        from openai import OpenAI
        llm_config = get_llm_config()
        
        self.client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config.get('base_url') or None
        )
        self.llm_model = llm_config['model']
    
    def retrieve_similar_exercises(self,
                                  bundle: List[int],
                                  concept_names: Dict[int, str]) -> List[Dict]:
        """RAG: retrieve similar exercises from problem_info."""
        bundle_names = [concept_names.get(cid, f"Concept_{cid}") for cid in bundle]
        retrieved = []
        for pid, p_info in self.problem_info.items():
            p_concepts = p_info.get('concepts', [])
            overlap = len(set(bundle_names) & set(p_concepts))
            if overlap > 0:
                retrieved.append({
                    'problem_id': pid,
                    'content': p_info.get('content', ''),
                    'concepts': p_concepts,
                    'overlap': overlap
                })
        retrieved.sort(key=lambda x: x['overlap'], reverse=True)
        return retrieved[:self.rag_top_k]
    
    def teacher_generate(self,
                        bundle: List[int],
                        concept_names: Dict[int, str],
                        retrieved_exercises: List[Dict],
                        mastery_levels: Dict[int, float],
                        bundle_selection_reason: str) -> Tuple[str, str, str]:
        """Teacher: generate exercise, answer, and explanation (J_t). Returns (exercise, answer, explanation)."""
        if not bundle_selection_reason or not bundle_selection_reason.strip():
            raise ValueError("bundle_selection_reason (j_t) is required and must be non-empty")
        bundle_names = [concept_names.get(cid, f"Concept_{cid}") for cid in bundle]
        examples_text = ""
        for i, ex in enumerate(retrieved_exercises[:3], 1):
            examples_text += f"\nExample {i}: {ex['content']}\n"
        
        has_reference = len(retrieved_exercises) > 0
        lang = get_prompt_lang(self.config)
        _, style_instruction = get_prompt(
            "teacher_style_ref" if has_reference else "teacher_style_free",
            lang=lang
        )
        reference_section = f'Reference examples:\n{examples_text}' if has_reference else ''
        messages = build_messages(
            "teacher",
            config=self.config,
            bundle_names=', '.join(bundle_names),
            bundle_selection_reason=bundle_selection_reason.strip(),
            mastery_levels=str(mastery_levels),
            reference_section=reference_section,
            style_instruction=style_instruction.strip()
        )
        
        try:
            self.logger.debug("Calling LLM (Teacher) to generate exercise...")
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                timeout=60
            )
            self.logger.debug("LLM (Teacher) response received")
            response = completion.choices[0].message.content.strip()
            exercise, answer, explanation = "", "", ""
            if "Exercise:" in response:
                rest = response.split("Exercise:")[1]
                if "Answer:" in rest:
                    exercise = rest.split("Answer:")[0].strip()
                    after_answer = rest.split("Answer:")[1]
                    if "Explanation:" in after_answer:
                        answer = after_answer.split("Explanation:")[0].strip()
                        explanation = after_answer.split("Explanation:")[1].strip().split("\n")[0].strip()
                    else:
                        answer = after_answer.strip()
                else:
                    exercise = rest.strip()
            if not exercise or not answer or not explanation:
                raise ValueError(
                    f"Teacher output incomplete: missing Exercise/Answer/Explanation. "
                    f"Got exercise={bool(exercise)}, answer={bool(answer)}, explanation={bool(explanation)}. "
                    f"Response preview: {response[:300]}..."
                )
            return exercise, answer, explanation
            
        except Exception as e:
            self.logger.error(f"Teacher generation error: {e}")
            raise RuntimeError(
                f"Failed to call LLM for Teacher generation: {e}. "
                f"Please check your network connection and LLM API configuration."
            ) from e
    
    def solver_solve(self, exercise: str) -> str:
        """Solver: solve exercise (blind)."""
        messages = build_messages("solver", config=self.config, exercise=exercise)
        
        try:
            self.logger.debug("Calling LLM (Solver) to solve exercise...")
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.solver_temperature,
                timeout=60
            )
            answer = completion.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            self.logger.error(f"Solver error: {e}")
            raise RuntimeError(
                f"Failed to call LLM for Solver: {e}. "
                f"Please check your network connection and LLM API configuration."
            ) from e
    
    def critic_validate(self,
                       exercise: str,
                       ref_answer: str,
                       sol_answer: str,
                       bundle: List[int],
                       concept_names: Dict[int, str],
                       bundle_selection_reason: str,
                       exercise_explanation: str) -> Tuple[bool, str]:
        """Critic: validate consistency, alignment, explanation, path alignment."""
        if not bundle_selection_reason or not bundle_selection_reason.strip():
            raise ValueError("bundle_selection_reason (j_t) is required for Critic")
        if not exercise_explanation or not exercise_explanation.strip():
            raise ValueError("exercise_explanation (J_t) is required for Critic")
        bundle_names = [concept_names.get(cid, f"Concept_{cid}") for cid in bundle]
        messages = build_messages(
            "critic",
            config=self.config,
            exercise=exercise,
            ref_answer=ref_answer,
            sol_answer=sol_answer,
            bundle_names=', '.join(bundle_names),
            bundle_selection_reason=bundle_selection_reason.strip(),
            exercise_explanation=exercise_explanation.strip()
        )
        
        try:
            self.logger.debug("Calling LLM (Critic) to validate exercise...")
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.1,
                timeout=60
            )
            response = completion.choices[0].message.content.strip()
            is_valid = "True" in response or "valid" in response.lower()
            reason = response.split("Reason:")[1].strip() if "Reason:" in response else ""
            
            return is_valid, reason
            
        except Exception as e:
            self.logger.error(f"Critic validation error: {e}")
            raise RuntimeError(
                f"Failed to call LLM for Critic validation: {e}. "
                f"Please check your network connection and LLM API configuration."
            ) from e
    
    def generate_exercise(self,
                         bundle: List[int],
                         concept_names: Dict[int, str],
                         mastery_levels: Dict[int, float],
                         bundle_selection_reason: str) -> Tuple[Optional[str], Optional[str], Optional[str], bool, bool, Optional[str]]:
        """Full AGA flow. Returns (exercise, answer, explanation, is_valid, retrieval_success, reference_preview). bundle_selection_reason (j_t) required."""
        if not bundle_selection_reason or not bundle_selection_reason.strip():
            raise ValueError("bundle_selection_reason (j_t) is required for generate_exercise")
        retrieved = self.retrieve_similar_exercises(bundle, concept_names)
        retrieval_success = len(retrieved) > 0
        reference_raw = retrieved[0]['content'] if retrieved else None
        reference_preview = None
        if reference_raw:
            ref_str = reference_raw.strip()
            if len(ref_str) > REFERENCE_EXERCISE_TRUNCATE_LEN:
                reference_preview = ref_str[:REFERENCE_EXERCISE_TRUNCATE_LEN] + "..."
            else:
                reference_preview = ref_str if ref_str else None
        for attempt in range(self.max_retries):
            self.logger.debug(f"AGA generation attempt {attempt + 1}/{self.max_retries}")
            try:
                exercise, ref_answer, exercise_explanation = self.teacher_generate(
                    bundle, concept_names, retrieved, mastery_levels,
                    bundle_selection_reason=bundle_selection_reason
                )
            except ValueError as e:
                self.logger.warning(f"Teacher parse error on attempt {attempt + 1}: {e}")
                continue
            
            if not exercise:
                self.logger.warning(f"Teacher returned empty exercise on attempt {attempt + 1}")
                continue
            
            sol_answer = self.solver_solve(exercise)
            
            try:
                is_valid, reason = self.critic_validate(
                    exercise, ref_answer, sol_answer, bundle, concept_names,
                    bundle_selection_reason=bundle_selection_reason,
                    exercise_explanation=exercise_explanation
                )
            except ValueError as e:
                self.logger.warning(f"Critic init error: {e}")
                continue
            
            if is_valid:
                self.logger.debug("AGA generation successful")
                return exercise, ref_answer, exercise_explanation, True, retrieval_success, reference_preview
            else:
                self.logger.debug(f"Validation failed: {reason}")
        
        raise RuntimeError(
            f"AGA generation failed after {self.max_retries} attempts. "
            f"Teacher generation or Critic validation failed. "
            f"Please check your LLM API configuration and network connection."
        )
