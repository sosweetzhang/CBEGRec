from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from prompts import build_messages, get_prompt_lang
from src.translator.ns_translator import NSTranslator
from src.utils.llm_client import call_llm


@dataclass(frozen=True)
class BundleCandidate:
    b_t: List[int]
    score: float
    U_t: float
    Coh_t: float
    Dist_t: float
    j_t: str = ""


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_bundle(bundle: Sequence[int]) -> List[int]:
    return sorted({int(c) for c in bundle})


class GoalOrientedConceptBundling:

    def __init__(
        self,
        kg_structure: Dict[str, List[str]],
        config: dict,
        logger: Any = None,
        llm_client: Any = None,
    ):
        self.logger = logger
        self.config = config or {}
        self.kg_structure = {str(k): [str(v) for v in (values or [])] for k, values in (kg_structure or {}).items()}
        bundling_config = self.config.get("concept_bundling", {})
        self.frontier_threshold = float(bundling_config.get("frontier_threshold", 0.6))
        self.max_concepts_per_bundle = int(bundling_config.get("max_concepts_per_bundle", 4))
        self.top_k_bundles = int(bundling_config.get("top_k_bundles", 5))
        self.weight_zpd = float(bundling_config.get("weight_zpd", 0.35))
        self.weight_cohesion = float(bundling_config.get("weight_cohesion", 0.20))
        self.weight_goal_proximity = float(bundling_config.get("weight_goal_proximity", 0.45))
        self.max_frontier_candidates = int(bundling_config.get("max_frontier_candidates", 30))
        self.llm_client = llm_client
        self.translator = NSTranslator()
        self.nodes = self._build_nodes()
        self.successors = self._build_successors()
        self.predecessors = self._build_predecessors()

    def _build_nodes(self) -> List[int]:
        nodes: Set[int] = set()
        for concept_id, prereqs in self.kg_structure.items():
            node = _to_int(concept_id)
            if node is not None:
                nodes.add(node)
            for prereq in prereqs:
                prereq_id = _to_int(prereq)
                if prereq_id is not None:
                    nodes.add(prereq_id)
        return sorted(nodes)

    def _build_successors(self) -> Dict[int, Set[int]]:
        successors: Dict[int, Set[int]] = {node: set() for node in self.nodes}
        for concept_id, prereqs in self.kg_structure.items():
            child = _to_int(concept_id)
            if child is None:
                continue
            for prereq in prereqs:
                parent = _to_int(prereq)
                if parent is None:
                    continue
                successors.setdefault(parent, set()).add(child)
                successors.setdefault(child, set())
        return successors

    def _build_predecessors(self) -> Dict[int, Set[int]]:
        predecessors: Dict[int, Set[int]] = {node: set() for node in self.nodes}
        for parent, children in self.successors.items():
            for child in children:
                predecessors.setdefault(child, set()).add(parent)
        return predecessors

    def _mastery(self, mastery_probs: Dict[int, float], concept_id: int) -> float:
        return float(mastery_probs.get(concept_id, 0.0))

    def identify_frontier(self, mastery_probs: Dict[int, float]) -> Set[int]:
        frontier: Set[int] = set()
        for concept_id in self.nodes:
            mastery = self._mastery(mastery_probs, concept_id)
            if mastery > self.frontier_threshold:
                continue
            prereqs = self.predecessors.get(concept_id, set())
            if all(self._mastery(mastery_probs, prereq) > self.frontier_threshold for prereq in prereqs):
                frontier.add(concept_id)
        return frontier

    def _candidate_neighborhood(self, frontier: Set[int]) -> List[int]:
        neighborhood: Set[int] = set(frontier)
        for node in frontier:
            neighborhood.update(self.successors.get(node, set()))
            neighborhood.update(self.predecessors.get(node, set()))
        return sorted(neighborhood)

    def _is_connected(self, bundle: Sequence[int], allowed: Set[int]) -> bool:
        bundle = list(dict.fromkeys(bundle))
        if len(bundle) <= 1:
            return True
        bundle_set = set(bundle)
        queue: deque[int] = deque([bundle[0]])
        visited = {bundle[0]}
        while queue:
            node = queue.popleft()
            neighbors = self.successors.get(node, set()) | self.predecessors.get(node, set())
            for neighbor in neighbors:
                if neighbor in bundle_set and neighbor not in visited and neighbor in allowed:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return visited == bundle_set

    def mine_preliminary_bundles(self, frontier: Set[int]) -> List[List[int]]:
        if not frontier:
            return []
        candidates = self._candidate_neighborhood(frontier)
        if len(candidates) > self.max_frontier_candidates:
            frontier_sorted = sorted(frontier)
            remaining = [c for c in candidates if c not in frontier_sorted]
            candidates = frontier_sorted + remaining[: self.max_frontier_candidates - len(frontier_sorted)]
        max_size = min(self.max_concepts_per_bundle, len(candidates))
        bundles: List[List[int]] = []
        allowed = set(candidates)
        for size in range(1, max_size + 1):
            for combo in combinations(candidates, size):
                combo_list = _normalize_bundle(combo)
                if combo_list and any(node in frontier for node in combo_list) and self._is_connected(combo_list, allowed):
                    bundles.append(combo_list)
        unique: List[List[int]] = []
        seen: Set[Tuple[int, ...]] = set()
        for bundle in bundles:
            key = tuple(bundle)
            if key not in seen:
                seen.add(key)
                unique.append(bundle)
        return unique

    def _entropy(self, p: float) -> float:
        p = max(1e-12, min(1 - 1e-12, p))
        return -(p * log(p) + (1 - p) * log(1 - p))

    def compute_zpd_utility(self, bundle: Sequence[int], mastery_probs: Dict[int, float]) -> float:
        if not bundle:
            return 0.0
        return sum(self._entropy(self._mastery(mastery_probs, cid)) for cid in bundle) / len(bundle)

    def compute_cohesion(self, bundle: Sequence[int]) -> float:
        bundle = _normalize_bundle(bundle)
        if len(bundle) <= 1:
            return 1.0
        edges = 0
        for parent in bundle:
            for child in self.successors.get(parent, set()):
                if child in bundle:
                    edges += 1
        max_edges = len(bundle) * (len(bundle) - 1)
        return edges / max_edges if max_edges else 0.0

    def _distance_to_goal(self, concept_id: int, goal: int) -> float:
        if concept_id == goal:
            return 0.0
        queue: deque[Tuple[int, int]] = deque([(concept_id, 0)])
        visited = {concept_id}
        while queue:
            node, dist = queue.popleft()
            for neighbor in self.successors.get(node, set()):
                if neighbor == goal:
                    return float(dist + 1)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        return float("inf")

    def compute_goal_distance(self, bundle: Sequence[int], goal: int) -> float:
        if not bundle:
            return float("inf")
        distances = [self._distance_to_goal(cid, goal) for cid in bundle]
        finite = [d for d in distances if d != float("inf")]
        return min(finite) if finite else float("inf")

    def score_bundle(self, bundle: Sequence[int], mastery_probs: Dict[int, float], goal: int) -> float:
        bundle = _normalize_bundle(bundle)
        utility = self.compute_zpd_utility(bundle, mastery_probs)
        cohesion = self.compute_cohesion(bundle)
        distance = self.compute_goal_distance(bundle, goal)
        if distance == float("inf"):
            distance = 100.0
        return self.weight_zpd * utility + self.weight_cohesion * cohesion - self.weight_goal_proximity * distance

    def select_top_k_bundles(self, bundles: Iterable[Sequence[int]], mastery_probs: Dict[int, float], goal: int) -> List[BundleCandidate]:
        scored = []
        for bundle in bundles:
            normalized = _normalize_bundle(bundle)
            scored.append(
                BundleCandidate(
                    b_t=normalized,
                    score=self.score_bundle(normalized, mastery_probs, goal),
                    U_t=self.compute_zpd_utility(normalized, mastery_probs),
                    Coh_t=self.compute_cohesion(normalized),
                    Dist_t=self.compute_goal_distance(normalized, goal),
                )
            )
        scored.sort(key=lambda item: (-item.score, len(item.b_t), item.b_t))
        return scored[: self.top_k_bundles]

    def _bundle_descriptions(self, bundles: Sequence[BundleCandidate], concept_names: Dict[int, str], mastery_probs: Dict[int, float]) -> str:
        lines = []
        for idx, bundle in enumerate(bundles, start=1):
            names = [concept_names.get(cid, f"Concept_{cid}") for cid in bundle.b_t]
            mastery = {concept_names.get(cid, f"Concept_{cid}"): round(mastery_probs.get(cid, 0.0), 4) for cid in bundle.b_t}
            lines.append(f"Bundle {idx}: {', '.join(names)} | mastery={mastery} | score={bundle.score:.4f}")
        return "\n".join(lines)

    def refine_with_llm(
        self,
        bundles: Sequence[BundleCandidate],
        mastery_probs: Dict[int, float],
        goal: int,
        concept_names: Dict[int, str],
        recent_bundles: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], str]:
        if not bundles:
            return [], ""
        goal_name = concept_names.get(goal, f"Concept_{goal}")
        profile_text = self.translator.generate_profile(mastery_probs, concept_names, goal)
        recent = recent_bundles or []
        if recent:
            diversity_hint = f"\nRecent bundles: {recent[:5]}.\n"
        else:
            diversity_hint = ""
        messages = build_messages(
            "bundle_select",
            config=self.config,
            diversity_hint=diversity_hint,
            profile_text=profile_text,
            goal_name=goal_name,
            bundle_descriptions=self._bundle_descriptions(bundles, concept_names, mastery_probs),
        )
        if self.llm_client is None:
            chosen = bundles[0]
            reason = f"Selected {', '.join(concept_names.get(cid, str(cid)) for cid in chosen.b_t)} by paper score."
            return chosen.b_t, reason
        response = call_llm(self.llm_client, "BundleSelection", messages, temperature=0.0)
        reason = response
        selected_index = 0
        lowered = response.lower()
        for idx in range(1, len(bundles) + 1):
            if f"bundle {idx}" in lowered or f"bundle {idx}:" in lowered:
                selected_index = idx - 1
                break
        elif_match = None
        if selected_index == 0 and "reason:" in lowered:
            reason = response.split("Reason:", 1)[-1].strip()
        if "reason:" in response.lower():
            reason = response.split("Reason:", 1)[-1].strip() if "Reason:" in response else response.split("reason:", 1)[-1].strip()
        selected = bundles[selected_index]
        return selected.b_t, reason

    def select_single_concept(self, mastery_probs: Dict[int, float], goal: int) -> Tuple[List[int], str]:
        frontier = self.identify_frontier(mastery_probs)
        if not frontier:
            return [], ""
        singletons = [[cid] for cid in sorted(frontier)]
        scored = self.select_top_k_bundles(singletons, mastery_probs, goal)
        if not scored:
            return [], ""
        best = scored[0]
        return best.b_t, f"Single-concept selection based on frontier and goal proximity: {best.b_t}"

    def plan_bundle(
        self,
        mastery_probs: Dict[int, float],
        goal: int,
        concept_names: Dict[int, str],
        recent_bundles: Optional[List[List[int]]] = None,
    ) -> Tuple[List[int], str, List[BundleCandidate]]:
        frontier = self.identify_frontier(mastery_probs)
        bundles = self.mine_preliminary_bundles(frontier)
        top = self.select_top_k_bundles(bundles, mastery_probs, goal)
        selected, reason = self.refine_with_llm(top, mastery_probs, goal, concept_names, recent_bundles=recent_bundles)
        return selected, reason, top
