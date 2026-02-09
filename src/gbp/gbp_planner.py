"""Goal-Oriented Bundle Planner (GBP)."""
import json
import re
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import get_llm_config
from prompts import build_messages, get_prompt_lang
from src.utils.logger import AppLogger
from src.translator.ns_translator import NSTranslator

class GBPPlanner:
    """Goal-oriented bundle planner."""
    
    def __init__(self, 
                 kg_structure: Dict[str, List[str]],
                 config: dict,
                 logger: AppLogger = None):
        self.kg_structure = kg_structure
        self.config = config
        self.logger = logger or AppLogger("GBP")
        self.translator = NSTranslator(
            threshold_low=config.get('ecge', {}).get('uncertainty_threshold_low', 0.3),
            threshold_high=config.get('ecge', {}).get('uncertainty_threshold_high', 0.7)
        )
        self.graph = self._build_graph()
        self.distances_to_goal = {}
        gbp_config = config.get('gbp', {})
        self.w1 = gbp_config.get('weight_zpd', 1.0)
        self.w2 = gbp_config.get('weight_cohesion', 0.5)
        self.w3 = gbp_config.get('weight_goal_proximity', 0.3)
        self.top_n = gbp_config.get('top_n_candidates', 5)
        self.frontier_threshold = gbp_config.get('frontier_threshold_high', 0.7)
        self.max_concepts_per_step = gbp_config.get('max_concepts_per_step', 15)
        self.max_bundles_per_step = gbp_config.get('max_bundles_per_step', 30)
        self.repetition_penalty_weight = gbp_config.get('repetition_penalty_weight', 2.0)
        self.repetition_lookback_steps = gbp_config.get('repetition_lookback_steps', 5)
        self.exact_match_penalty = gbp_config.get('exact_match_penalty', 5.0)
        from src.utils.disable_proxy import ensure_no_proxy
        ensure_no_proxy()
        llm_config = get_llm_config()
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=llm_config['api_key'],
            base_url=llm_config.get('base_url') or None
        )
        self.llm_model = llm_config['model']
    
    def _build_graph(self) -> nx.DiGraph:
        """Build NetworkX digraph from KG."""
        G = nx.DiGraph()
        for concept, prereqs in self.kg_structure.items():
            G.add_node(concept)
            for prereq in prereqs:
                G.add_edge(prereq, concept)
        return G
    
    def precompute_distances(self, goal: str):
        """Precompute shortest-path distances from all nodes to goal."""
        if goal not in self.graph:
            self.logger.warning(f"Goal {goal} not in graph")
            return
        
        try:
            lengths = nx.single_target_shortest_path_length(self.graph, goal)
            if not isinstance(lengths, dict):
                lengths = dict(lengths)
            self.distances_to_goal = {str(k): v for k, v in lengths.items()}
        except nx.NetworkXNoPath:
            self.logger.warning(f"No path to goal {goal}")
            self.distances_to_goal = {str(n): float('inf') for n in self.graph.nodes()}
    
    def identify_frontier(self,
                         mastery_probs: Dict[int, float],
                         concept_names: Dict[int, str]) -> Set[int]:
        """
        3.5.2 Frontier Masking
        Identify frontier: prereqs mastered but self not mastered.
        """
        frontier = set()
        
        for concept_id, prob in mastery_probs.items():
            concept_str = str(concept_id)
            
            if concept_str not in self.graph:
                continue
            
            prereqs = self.kg_structure.get(concept_str, [])
            all_prereqs_mastered = True
            for prereq_str in prereqs:
                prereq_id = int(prereq_str) if prereq_str.isdigit() else None
                if prereq_id and prereq_id in mastery_probs:
                    if mastery_probs[prereq_id] <= self.frontier_threshold:
                        all_prereqs_mastered = False
                        break
                else:
                    all_prereqs_mastered = False
                    break
            if all_prereqs_mastered and prob <= self.frontier_threshold:
                frontier.add(concept_id)
        
        self.logger.info(f"Identified {len(frontier)} frontier concepts")
        return frontier
    
    def mine_bundles(self, frontier: Set[int], max_size: int = 3) -> List[List[int]]:
        """
        3.5.2 Candidate Mining
        Mine connected subgraphs on frontier and neighbors as candidate bundles.
        """
        bundles = []
        max_bundles = 200
        frontier_nodes = {str(cid) for cid in frontier}
        neighbors = set()
        for node in frontier_nodes:
            if node in self.graph:
                neighbors.update(self.graph.successors(node))
                neighbors.update(self.graph.predecessors(node))
        all_candidates = list(frontier) + [int(n) for n in neighbors if n.isdigit()]
        max_concepts = getattr(self, 'max_concepts_per_step', 15)
        if len(all_candidates) > max_concepts:
            frontier_limit = max(1, int(max_concepts * 0.8))
            neighbor_limit = max_concepts - frontier_limit
            all_candidates = list(frontier)[:frontier_limit] + [int(n) for n in neighbors if n.isdigit()][:neighbor_limit]
            self.logger.info(f"Limited candidates from {len(frontier) + len(neighbors)} to {len(all_candidates)}")
        from itertools import combinations
        max_bundles_limit = getattr(self, 'max_bundles_per_step', 30)
        for size in range(1, min(max_size + 1, len(all_candidates) + 1)):
            if len(bundles) >= max_bundles_limit:
                break
            for combo in combinations(all_candidates, size):
                if len(bundles) >= max_bundles_limit:
                    break
                if self._is_connected_bundle([str(c) for c in combo]):
                    bundles.append(list(combo))
        
        self.logger.info(f"Mined {len(bundles)} candidate bundles")
        return bundles
    
    def _is_connected_bundle(self, bundle: List[str]) -> bool:
        """Check if bundle is connected (simplified)."""
        if len(bundle) <= 1:
            return True
        subgraph = self.graph.subgraph(bundle)
        return nx.is_weakly_connected(subgraph) or len(bundle) <= 2
    
    def calculate_cohesion(self, bundle: List[int]) -> float:
        """Compute bundle cohesion."""
        if len(bundle) <= 1:
            return 1.0
        
        bundle_str = [str(c) for c in bundle]
        subgraph = self.graph.subgraph(bundle_str)
        num_edges = subgraph.number_of_edges()
        num_nodes = len(bundle)
        max_edges = num_nodes * (num_nodes - 1)
        
        if max_edges == 0:
            return 1.0
        
        return num_edges / max_edges if max_edges > 0 else 0.0
    
    def _compute_repetition_penalty(self, bundle: List[int], recent_bundles: List[List[int]]) -> float:
        """Penalty for overlap with recent bundles; exact match gets extra penalty."""
        if not recent_bundles or not bundle:
            return 0.0
        bundle_set = frozenset(bundle)
        penalty = 0.0
        for i, past in enumerate(recent_bundles):
            past_set = set(past)
            if not past_set:
                continue
            if frozenset(past) == bundle_set:
                decay = 1.0 / (1.0 + i * 0.4)
                return self.exact_match_penalty * decay
            overlap = len(bundle_set & past_set) / len(bundle_set)
            decay = 1.0 / (1.0 + i * 0.3)
            penalty += overlap * decay
        return min(3.0, penalty)

    def score_bundle(self,
                    bundle: List[int],
                    uncertainties: Dict[int, float],
                    goal: int,
                    recent_bundles: Optional[List[List[int]]] = None) -> float:
        """Multi-objective navigation score including repetition penalty."""
        avg_uncertainty = np.mean([uncertainties.get(c, 0.0) for c in bundle])
        min_dist = min([self.distances_to_goal.get(str(c), float('inf')) 
                       for c in bundle])
        if min_dist == float('inf'):
            min_dist = 100
        cohesion = self.calculate_cohesion(bundle)
        score = (self.w1 * avg_uncertainty + 
                self.w2 * cohesion - 
                self.w3 * min_dist)
        
        if recent_bundles and self.repetition_penalty_weight > 0:
            penalty = self._compute_repetition_penalty(bundle, recent_bundles)
            score -= self.repetition_penalty_weight * penalty
        
        return score
    
    def select_top_bundles(self,
                          bundles: List[List[int]],
                          uncertainties: Dict[int, float],
                          goal: int,
                          recent_bundles: Optional[List[List[int]]] = None) -> List[List[int]]:
        """Select top-N bundles with repetition penalty."""
        scored_bundles = [
            (b, self.score_bundle(b, uncertainties, goal, recent_bundles))
            for b in bundles
        ]
        scored_bundles.sort(key=lambda x: x[1], reverse=True)
        
        top_bundles = [b for b, _ in scored_bundles[:self.top_n]]
        if top_bundles:
            top_scores = [s for _, s in scored_bundles[:self.top_n]]
            self.logger.debug(
                f"Top {len(top_bundles)} bundles scores: {[f'{s:.3f}' for s in top_scores]}"
            )
        
        return top_bundles
    
    def llm_select_final_bundle(self,
                                top_bundles: List[List[int]],
                                profile_text: str,
                                goal_name: str,
                                concept_names: Dict[int, str],
                                recent_bundle_names: Optional[List[str]] = None) -> Tuple[List[int], str]:
        """LLM selects final bundle from top-N. Returns (bundle, bundle_selection_reason)."""
        bundle_descriptions = []
        for i, bundle in enumerate(top_bundles):
            bundle_text = self.translator.translate_bundle_to_text(
                bundle, {}, concept_names
            )
            bundle_descriptions.append(f"Bundle {i+1}: {bundle_text}")
        
        if recent_bundle_names:
            hint = ", ".join(recent_bundle_names[:5])
            diversity_hint = f"\n[Important] Recent bundles used: {hint}. Prefer a different bundle to avoid repetition.\n"
        else:
            diversity_hint = ""
        messages = build_messages(
            "bundle_select",
            config=self.config,
            diversity_hint=diversity_hint,
            profile_text=profile_text,
            goal_name=goal_name,
            bundle_descriptions=chr(10).join(bundle_descriptions)
        )
        
        try:
            self.logger.debug(f"Calling LLM to select bundle from {len(top_bundles)} candidates...")
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.3,
                timeout=60
            )
            self.logger.debug("LLM response received")
            response = completion.choices[0].message.content.strip()
            reason = ""
            if "Reason:" in response:
                reason = response.split("Reason:")[-1].strip().split("\n")[0].strip()
            elif "reason:" in response.lower():
                reason = response.split("reason:")[-1].strip().split("\n")[0].strip()
            if not reason:
                reason = response.strip()[:200]
            bundle_match = re.search(r'Bundle\s*(\d+)', response, re.IGNORECASE)
            if bundle_match:
                bundle_num = int(bundle_match.group(1)) - 1
                if 0 <= bundle_num < len(top_bundles):
                    self.logger.debug(f"LLM selected Bundle {bundle_num + 1}")
                    return top_bundles[bundle_num], reason
            
            num_match = re.search(r'\b(\d+)\b', response)
            if num_match:
                bundle_num = int(num_match.group(1)) - 1
                if 0 <= bundle_num < len(top_bundles):
                    self.logger.debug(f"LLM selected Bundle {bundle_num + 1} (from number)")
                    return top_bundles[bundle_num], reason
            
            raise ValueError(
                f"LLM selection failed: Could not parse bundle selection from response: {response[:200]}. "
                f"Expected format: 'Bundle X' or just a number."
            )
            
        except Exception as e:
            self.logger.error(f"LLM selection error: {e}")
            raise RuntimeError(
                f"Failed to call LLM for bundle selection: {e}. "
                f"Please check your network connection and LLM API configuration."
            ) from e
    
    def plan_bundle(self,
                   mastery_probs: Dict[int, float],
                   uncertainties: Dict[int, float],
                   goal: int,
                   concept_names: Dict[int, str],
                   recent_bundles: Optional[List[List[int]]] = None) -> Tuple[List[int], str]:
        """Full bundle planning. Returns (bundle, bundle_selection_reason)."""
        goal_str = str(goal)
        if goal_str not in self.graph:
            goal_name = concept_names.get(goal, str(goal))
            found = False
            for node in self.graph.nodes():
                if node == goal_name or str(node) == goal_name:
                    goal_str = str(node)
                    found = True
                    break
            if not found:
                self.logger.warning(f"Goal {goal} ({goal_name}) not in graph, distances may be invalid")
        
        self.precompute_distances(goal_str)
        frontier = self.identify_frontier(mastery_probs, concept_names)
        if not frontier:
            self.logger.warning("    No frontier concepts found")
            return [], ""
        bundles = self.mine_bundles(frontier)
        if not bundles:
            self.logger.warning("    No candidate bundles found")
            return [], ""
        
        lookback = getattr(self, 'repetition_lookback_steps', 4)
        recent = (recent_bundles or [])[-lookback:]
        top_bundles = self.select_top_bundles(bundles, uncertainties, goal, recent)
        profile_text = self.translator.generate_profile(
            mastery_probs, concept_names, goal
        )
        goal_name = concept_names.get(goal, f"Concept_{goal}")
        recent_names = None
        if recent:
            recent_names = [
                self.translator.translate_bundle_to_text(b, {}, concept_names)
                for b in recent
            ]
        final_bundle, bundle_reason = self.llm_select_final_bundle(
            top_bundles, profile_text, goal_name, concept_names, recent_bundle_names=recent_names
        )
        return final_bundle, bundle_reason
