from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prompts import build_messages, get_prompt, get_prompt_lang
from src.utils.llm_client import call_llm


@dataclass
class GeneratedExercise:
    e_t: str
    a_ref_t: str
    J_t: str
    a_sol_t: str
    valid_t: bool
    retry_count: int
    retrieval_used: bool
    reference_preview: Optional[str]
    feedback: str = ""


def parse_teacher_output(response: str) -> Tuple[str, str, str]:
    exercise = ""
    answer = ""
    explanation = ""
    if "Exercise:" in response:
        after_exercise = response.split("Exercise:", 1)[1]
        if "Answer:" in after_exercise:
            exercise = after_exercise.split("Answer:", 1)[0].strip()
            after_answer = after_exercise.split("Answer:", 1)[1]
            if "Explanation:" in after_answer:
                answer = after_answer.split("Explanation:", 1)[0].strip()
                explanation = after_answer.split("Explanation:", 1)[1].strip()
            else:
                answer = after_answer.strip()
    if not exercise or not answer or not explanation:
        raise ValueError("Teacher output incomplete: missing Exercise/Answer/Explanation.")
    return exercise, answer, explanation


def parse_critic_validation(response: str) -> Tuple[bool, str]:
    valid_match = re.search(r"Valid:\s*(True|False)", response, flags=re.IGNORECASE)
    if valid_match:
        valid = valid_match.group(1).lower() == "true"
    else:
        lowered = response.lower()
        if "valid: true" in lowered:
            valid = True
        elif "valid: false" in lowered:
            valid = False
        else:
            valid = False
    reason = ""
    reason_match = re.search(r"Reason:\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
    if reason_match:
        reason = reason_match.group(1).strip().splitlines()[0].strip()
    return valid, reason


class PathAwareExerciseGeneration:

    def __init__(
        self,
        problem_info: Dict,
        config: dict,
        llm_client: Any = None,
        logger: Any = None,
    ):
        self.problem_info = problem_info or {}
        self.config = config or {}
        self.logger = logger
        generation_config = self.config.get("exercise_generation", {})
        self.max_retries = int(generation_config.get("max_retries", 3))
        self.rag_top_k = int(generation_config.get("rag_top_k", 5))
        self.teacher_temperature = float(generation_config.get("teacher_temperature", 0.0))
        self.solver_temperature = float(generation_config.get("solver_temperature", 0.0))
        self.critic_temperature = float(generation_config.get("critic_temperature", 0.0))
        self.llm_client = llm_client

    def _concept_names(self, bundle: Sequence[int], concept_names: Dict[int, str]) -> List[str]:
        return [concept_names.get(cid, f"Concept_{cid}") for cid in bundle]

    def _format_problem_content(self, problem: Dict) -> str:
        content = problem.get("content", "")
        options = problem.get("options")
        if isinstance(options, dict) and options:
            option_lines = [f"{key}. {value}" for key, value in options.items()]
            content = "\n".join([content, *option_lines])
        return content

    def retrieve_reference_exercises(self, bundle: Sequence[int], concept_names: Dict[int, str]) -> List[Dict]:
        bundle_names = set(self._concept_names(bundle, concept_names))
        retrieved = []
        for problem_id, problem in self.problem_info.items():
            p_concepts = set(problem.get("concepts", []))
            overlap = len(bundle_names & p_concepts)
            if overlap > 0:
                retrieved.append(
                    {
                        "problem_id": problem_id,
                        "content": self._format_problem_content(problem),
                        "concepts": list(p_concepts),
                        "overlap": overlap,
                    }
                )
        retrieved.sort(key=lambda item: (-item["overlap"], str(item["problem_id"])))
        return retrieved[: self.rag_top_k]

    def _reference_section(self, retrieved: List[Dict]) -> str:
        if not retrieved:
            return ""
        lines = ["Reference examples:"]
        for idx, item in enumerate(retrieved[:3], start=1):
            lines.append(f"Example {idx}: {item['content']}")
        return "\n".join(lines)

    def teacher(
        self,
        b_t: Sequence[int],
        j_t: str,
        h_t: Dict[int, float],
        concept_names: Dict[int, str],
        retrieved: Optional[List[Dict]] = None,
        critic_feedback: str = "",
    ) -> Tuple[str, str, str]:
        bundle_names = ", ".join(self._concept_names(b_t, concept_names))
        lang = get_prompt_lang(self.config)
        style_key = "teacher_style_ref" if retrieved else "teacher_style_free"
        _, style_instruction = get_prompt(style_key, lang=lang)
        messages = build_messages(
            "teacher",
            config=self.config,
            bundle_names=bundle_names,
            j_t=j_t,
            h_t=h_t,
            reference_section=self._reference_section(retrieved or []),
            style_instruction=style_instruction.strip(),
            critic_feedback=f"Previous critic feedback: {critic_feedback}" if critic_feedback else "",
        )
        response = call_llm(self.llm_client, "Teacher", messages, temperature=self.teacher_temperature)
        return parse_teacher_output(response)

    def solver(self, e_t: str) -> str:
        messages = build_messages("solver", config=self.config, e_t=e_t)
        return call_llm(self.llm_client, "Solver", messages, temperature=self.solver_temperature)

    def critic(
        self,
        e_t: str,
        a_ref_t: str,
        a_sol_t: str,
        b_t: Sequence[int],
        j_t: str,
        J_t: str,
        concept_names: Dict[int, str],
    ) -> Tuple[bool, str]:
        bundle_names = ", ".join(self._concept_names(b_t, concept_names))
        messages = build_messages(
            "critic",
            config=self.config,
            e_t=e_t,
            a_ref_t=a_ref_t,
            a_sol_t=a_sol_t,
            bundle_names=bundle_names,
            j_t=j_t,
            J_t=J_t,
        )
        response = call_llm(self.llm_client, "Critic", messages, temperature=self.critic_temperature)
        return parse_critic_validation(response)

    def generate_exercise(
        self,
        b_t: Sequence[int],
        concept_names: Dict[int, str],
        h_t: Dict[int, float],
        j_t: str,
        recent_bundles: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        if not j_t or not j_t.strip():
            raise ValueError("j_t is required for exercise generation")
        retrieved = self.retrieve_reference_exercises(b_t, concept_names)
        retrieval_used = bool(retrieved)
        reference_preview = None
        if retrieved:
            reference_preview = retrieved[0]["content"][:200]
        feedback = ""
        for retry_index in range(1, self.max_retries + 1):
            e_t, a_ref_t, J_t = self.teacher(
                b_t=b_t,
                j_t=j_t,
                h_t=h_t,
                concept_names=concept_names,
                retrieved=retrieved if retrieval_used else None,
                critic_feedback=feedback,
            )
            a_sol_t = self.solver(e_t)
            valid_t, reason = self.critic(
                e_t=e_t,
                a_ref_t=a_ref_t,
                a_sol_t=a_sol_t,
                b_t=b_t,
                j_t=j_t,
                J_t=J_t,
                concept_names=concept_names,
            )
            if valid_t:
                return {
                    "e_t": e_t,
                    "a_ref_t": a_ref_t,
                    "J_t": J_t,
                    "a_sol_t": a_sol_t,
                    "valid_t": True,
                    "retry_count": retry_index,
                    "retrieval_used": retrieval_used,
                    "reference_preview": reference_preview,
                    "feedback": reason,
                }
            feedback = reason or "Critic rejected the exercise."
        raise RuntimeError(f"Exercise generation failed after {self.max_retries} retries: {feedback}")
