from __future__ import annotations

import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from prompts import build_messages, get_prompt_lang
from src.utils.llm_client import call_llm


def _to_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_dag(kg: Dict[str, List[str]]) -> bool:
    graph = defaultdict(list)
    indegree = defaultdict(int)
    nodes: Set[str] = set()
    for concept, prereqs in kg.items():
        nodes.add(str(concept))
        for prereq in prereqs or []:
            nodes.add(str(prereq))
            graph[str(prereq)].append(str(concept))
            indegree[str(concept)] += 1
            indegree.setdefault(str(prereq), 0)
    queue = deque([node for node in nodes if indegree.get(node, 0) == 0])
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for neighbor in graph.get(node, []):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    return visited == len(nodes)


def _detect_cycle_path(kg: Dict[str, List[str]]) -> Optional[List[str]]:
    graph = defaultdict(list)
    nodes: Set[str] = set()
    for concept, prereqs in kg.items():
        concept = str(concept)
        nodes.add(concept)
        for prereq in prereqs or []:
            prereq = str(prereq)
            nodes.add(prereq)
            graph[prereq].append(concept)

    visited: Set[str] = set()
    stack: Set[str] = set()
    parent: Dict[str, str] = {}

    def dfs(node: str) -> Optional[List[str]]:
        visited.add(node)
        stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                parent[neighbor] = node
                found = dfs(neighbor)
                if found:
                    return found
            elif neighbor in stack:
                cycle = [neighbor]
                cur = node
                while cur != neighbor:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(neighbor)
                cycle.reverse()
                return cycle
        stack.remove(node)
        return None

    for node in nodes:
        if node not in visited:
            found = dfs(node)
            if found:
                return found
    return None


def apply_topological_pruning(
    kg_structure: Dict[str, List[str]],
    transition_probabilities: Dict[Tuple[str, str], float],
) -> Tuple[Dict[str, List[str]], List[Tuple[str, str]]]:
    pruned = {str(concept): [str(prereq) for prereq in (prereqs or [])] for concept, prereqs in kg_structure.items()}
    removed_edges: List[Tuple[str, str]] = []

    while True:
        cycle = _detect_cycle_path(pruned)
        if not cycle:
            break
        cycle_edges = []
        for idx in range(len(cycle) - 1):
            prereq = cycle[idx]
            concept = cycle[idx + 1]
            probability = float(transition_probabilities.get((prereq, concept), 0.0))
            cycle_edges.append((probability, prereq, concept))
        if not cycle_edges:
            break
        _, prereq_to_remove, concept_to_remove = min(cycle_edges, key=lambda item: (item[0], item[1], item[2]))
        if prereq_to_remove in pruned.get(concept_to_remove, []):
            pruned[concept_to_remove].remove(prereq_to_remove)
            removed_edges.append((prereq_to_remove, concept_to_remove))
        else:
            break
    return pruned, removed_edges


def build_transition_graph(
    records: Sequence[Tuple[int, List[int], List[int]]],
    problem_info: Dict[str, Dict],
    concept2id: Dict[str, int],
) -> Dict[str, List[str]]:
    concept_counts = defaultdict(lambda: defaultdict(int))
    for _, question_ids, _ in records:
        concepts_per_position: List[List[int]] = []
        for qid in question_ids:
            info = problem_info.get(str(qid), {})
            concepts = []
            for concept_name in info.get("concepts", []):
                cid = concept2id.get(concept_name)
                if cid is not None:
                    concepts.append(int(cid))
            concepts_per_position.append(concepts)
        for idx in range(len(concepts_per_position) - 1):
            current = concepts_per_position[idx]
            nxt = concepts_per_position[idx + 1]
            for src in current:
                for dst in nxt:
                    if src != dst:
                        concept_counts[str(src)][str(dst)] += 1

    kg_structure: Dict[str, List[str]] = {str(cid): [] for cid in concept2id.values()}
    for src, targets in concept_counts.items():
        total = sum(targets.values())
        if total <= 0:
            continue
        for dst, count in targets.items():
            prob = count / total
            if prob > 0:
                kg_structure.setdefault(dst, [])
                kg_structure[dst].append(src)
    for concept in list(kg_structure):
        kg_structure[concept] = sorted(set(kg_structure[concept]), key=lambda x: int(x) if x.isdigit() else x)
    return kg_structure


class HybridKGConstruction:

    def __init__(self, config: dict, logger: Any = None, llm_client: Any = None):
        self.config = config or {}
        self.logger = logger
        self.llm_client = llm_client
        kg_config = self.config.get("kg_construction", {})
        self.llm_extraction_enabled = bool(kg_config.get("llm_extraction_enabled", True))
        self.data_validation_enabled = bool(kg_config.get("data_validation_enabled", True))
        self.use_separate_llm = bool(kg_config.get("use_separate_llm", True))
        self.significance_threshold = float(kg_config.get("significance_threshold", 0.1))

    def load_records(self, records_path: str) -> List[Tuple[int, List[int], List[int]]]:
        records = []
        with open(records_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        idx = 0
        while idx + 2 < len(lines):
            try:
                seq_len = int(lines[idx])
                question_ids = [int(x) for x in lines[idx + 1].split()]
                answers = [int(x) for x in lines[idx + 2].split()]
                if len(question_ids) == len(answers) == seq_len:
                    records.append((seq_len, question_ids, answers))
            except ValueError:
                pass
            idx += 3
        return records

    def build_transition_probabilities(
        self,
        records: Sequence[Tuple[int, List[int], List[int]]],
        problem_info: Dict[str, Dict],
        concept2id: Dict[str, int],
    ) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], float]]:
        counts = defaultdict(lambda: defaultdict(int))
        for _, question_ids, _ in records:
            concepts_sequence: List[List[int]] = []
            for qid in question_ids:
                info = problem_info.get(str(qid), {})
                concept_ids = []
                for concept_name in info.get("concepts", []):
                    cid = concept2id.get(concept_name)
                    if cid is not None:
                        concept_ids.append(int(cid))
                concepts_sequence.append(concept_ids)
            for idx in range(len(concepts_sequence) - 1):
                current = concepts_sequence[idx]
                nxt = concepts_sequence[idx + 1]
                for src in current:
                    for dst in nxt:
                        if src == dst:
                            continue
                        counts[str(src)][str(dst)] += 1
        kg_structure: Dict[str, List[str]] = {str(cid): [] for cid in concept2id.values()}
        transition_probabilities: Dict[Tuple[str, str], float] = {}
        for src, targets in counts.items():
            total = sum(targets.values())
            if total <= 0:
                continue
            for dst, count in targets.items():
                transition_probabilities[(src, dst)] = count / total
                kg_structure.setdefault(dst, [])
                kg_structure[dst].append(src)
        for concept in kg_structure:
            kg_structure[concept] = sorted(set(kg_structure[concept]), key=lambda x: int(x) if x.isdigit() else x)
        return kg_structure, transition_probabilities

    def semantic_refinement(
        self,
        kg_structure: Dict[str, List[str]],
        concepts: Dict[str, int],
        transition_probabilities: Dict[Tuple[str, str], float] = None,
    ) -> Dict[str, List[str]]:
        if self.llm_client is None:
            return kg_structure
        transition_probabilities = transition_probabilities or {}
        concept_lines = [f'"{name}": {cid}' for name, cid in concepts.items()]
        transition_edges = []
        for concept, prereqs in kg_structure.items():
            for prereq in prereqs:
                probability = transition_probabilities.get((str(prereq), str(concept)))
                if probability is None:
                    transition_edges.append(f"{prereq} -> {concept}")
                else:
                    transition_edges.append(f"{prereq} -> {concept} ({probability:.4f})")
        messages = build_messages(
            "kg_refinement",
            config=self.config,
            concepts_json="\n".join(concept_lines),
            transition_edges="\n".join(transition_edges),
        )
        response = call_llm(self.llm_client, "KGRefinement", messages, temperature=0.0)
        refined = {str(concept): list(prereqs) for concept, prereqs in kg_structure.items()}
        added_edges = re_find_tuples(response, prefix="Added edges")
        removed_edges = re_find_tuples(response, prefix="Removed edges")
        for src, dst in removed_edges:
            if dst in refined and src in refined[dst]:
                refined[dst].remove(src)
        for src, dst in added_edges:
            refined.setdefault(dst, [])
            if src not in refined[dst]:
                refined[dst].append(src)
        for concept in refined:
            refined[concept] = sorted(set(refined[concept]), key=lambda x: int(x) if x.isdigit() else x)
        return refined

    def construct_hybrid_kg(
        self,
        text_data: Sequence[str],
        records_path: str,
        problem_info_path: str,
        concept2id_path: str,
        existing_concepts: Dict[str, int] = None,
    ) -> Dict[str, List[str]]:
        with open(problem_info_path, "r", encoding="utf-8") as f:
            problem_info = json.load(f)
        concept2id = {}
        with open(concept2id_path, "r", encoding="utf-8") as f:
            for line in f:
                if "," in line:
                    key, value = line.strip().split(",", 1)
                    concept2id[key.strip('"')] = int(value.strip())
        records = self.load_records(records_path)
        kg_structure, transition_probabilities = self.build_transition_probabilities(records, problem_info, concept2id)
        if self.llm_extraction_enabled and self.llm_client is not None:
            refined = self.semantic_refinement(kg_structure, concept2id, transition_probabilities)
        else:
            refined = kg_structure
        pruned, _ = apply_topological_pruning(refined, transition_probabilities)
        return pruned


def re_find_tuples(response: str, prefix: str) -> List[Tuple[str, str]]:
    pattern = rf"{re.escape(prefix)}:\s*(.*)"
    match = re.search(pattern, response, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    tuples = re.findall(r"\(([^,]+),\s*([^)]+)\)", match.group(1))
    cleaned = []
    for left, right in tuples:
        cleaned.append((left.strip().strip('"').strip("'"), right.strip().strip('"').strip("'")))
    return cleaned
