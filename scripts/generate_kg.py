"""Generate Knowledge Graph: LLM extraction + data-driven validation, output DAG."""
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Set
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from config import load_config
from src.kg.hybrid_kg_construction import HybridKGConstruction
from src.utils.logger import AppLogger

DATASETS = ["PHP", "Mechanical_Physics", "Logistics"]


def load_concept2id(concept2id_path: str) -> dict:
    """Load concept2id mapping."""
    concept2id = {}
    with open(concept2id_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                parts = line.split(',', 1)
                concept_name = parts[0].strip().strip('"')
                concept_id = int(parts[1].strip())
                concept2id[concept_name] = concept_id
    return concept2id


def check_dag(kg: dict) -> Tuple[bool, List[List[int]]]:
    """Check if KG is DAG (DFS cycle detection). Returns (is_dag, cycles)."""
    graph = defaultdict(list)
    all_nodes = set()
    
    for concept_id, prerequisites in kg.items():
        concept_id_int = int(concept_id)
        all_nodes.add(concept_id_int)
        for prereq_id in prerequisites:
            prereq_id_int = int(prereq_id)
            all_nodes.add(prereq_id_int)
            graph[prereq_id_int].append(concept_id_int)
    
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in all_nodes}
    cycles = []
    path = []
    
    def dfs(node):
        if color[node] == GRAY:
            try:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
            except ValueError:
                pass
            return
        if color[node] == BLACK:
            return
        
        color[node] = GRAY
        path.append(node)
        
        for neighbor in graph.get(node, []):
            if neighbor in all_nodes:
                dfs(neighbor)
        
        path.pop()
        color[node] = BLACK
    
    for node in all_nodes:
        if color[node] == WHITE:
            dfs(node)
    
    return len(cycles) == 0, cycles


def remove_cycles(kg: Dict[str, List[str]]) -> Tuple[Dict[str, List[str]], int]:
    """Remove cycles from KG to get DAG (greedy MFAS). Returns (kg_dag, removed_count).
    """
    import copy
    kg_fixed = copy.deepcopy(kg)
    removed_count = 0
    max_iterations = 100
    
    for iteration in range(max_iterations):
        is_dag, cycles = check_dag(kg_fixed)
        if is_dag:
            break
        
        edge_cycle_count = defaultdict(int)
        for cycle in cycles:
            for i in range(len(cycle) - 1):
                from_node = cycle[i]
                to_node = cycle[i + 1]
                edge_cycle_count[(from_node, to_node)] += 1
        
        if not edge_cycle_count:
            break
        
        worst_edge = max(edge_cycle_count.items(), key=lambda x: x[1])
        from_node, to_node = worst_edge[0]
        to_node_str = str(to_node)
        from_node_str = str(from_node)
        
        if to_node_str in kg_fixed and from_node_str in kg_fixed[to_node_str]:
            kg_fixed[to_node_str].remove(from_node_str)
            removed_count += 1
    
    return kg_fixed, removed_count


def validate_kg_structure(kg: dict, concept2id: dict) -> Tuple[bool, List[str]]:
    """Validate KG structure."""
    errors = []
    all_concept_ids = set(concept2id.values())
    
    for concept_id_str, prerequisites in kg.items():
        try:
            concept_id = int(concept_id_str)
            if concept_id not in all_concept_ids:
                errors.append(f"Concept ID {concept_id} not in concept2id")
            
            for prereq_id_str in prerequisites:
                try:
                    prereq_id = int(prereq_id_str)
                    if prereq_id not in all_concept_ids:
                        errors.append(f"Prerequisite ID {prereq_id} not in concept2id")
                except ValueError:
                    errors.append(f"Invalid prerequisite ID: {prereq_id_str}")
        except ValueError:
            errors.append(f"Invalid concept ID: {concept_id_str}")
    kg_concept_ids = {int(k) for k in kg.keys()}
    missing_concepts = all_concept_ids - kg_concept_ids
    if missing_concepts:
        errors.append(f"Missing {len(missing_concepts)} concepts in KG")
        for missing_id in missing_concepts:
            kg[str(missing_id)] = []
    
    return len(errors) == 0, errors


def generate_kg_for_domain(domain: str, logger: AppLogger) -> Dict[str, List[str]]:
    """Generate KG for one dataset."""
    logger.info(f"{'='*60}")
    logger.info(f"Generating KG: {domain}")
    logger.info(f"{'='*60}")
    config = load_config(domain=domain)
    data_config = config['data']
    base_path = Path(data_config['base_path']) / domain
    
    records_path = base_path / data_config['records_file']
    problem_info_path = base_path / data_config['problem_info_file']
    concept2id_path = base_path / data_config['concept2id_file']
    kg_path = base_path / data_config['kg_structure_file']
    
    logger.info(f"Data path: {base_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Records file not found: {records_path}")
    if not problem_info_path.exists():
        raise FileNotFoundError(f"Problem info file not found: {problem_info_path}")
    if not concept2id_path.exists():
        raise FileNotFoundError(f"Concept2id file not found: {concept2id_path}")
    
    logger.info("Loading concept mapping...")
    concept2id = load_concept2id(str(concept2id_path))
    logger.info(f"Loaded {len(concept2id)} concepts")
    
    logger.info("Initializing hybrid KG builder...")
    kg_builder = HybridKGConstruction(config, logger)
    concepts_list = [f'"{name}": {cid}' for name, cid in concept2id.items()]
    
    logger.info("Building hybrid KG...")
    logger.info("Step 1: LLM extraction...")
    
    kg_structure = kg_builder.construct_hybrid_kg(
        text_data=concepts_list,
        records_path=str(records_path),
        problem_info_path=str(problem_info_path),
        concept2id_path=str(concept2id_path),
        existing_concepts=concept2id
    )
    
    if not kg_structure:
        raise RuntimeError("KG construction returned empty result")
    
    logger.info("Step 2: Validating KG...")
    is_valid, errors = validate_kg_structure(kg_structure, concept2id)
    if errors:
        logger.warning(f"Found {len(errors)} issues:")
        for error in errors[:5]:
            logger.warning(f"  - {error}")
    
    logger.info("Step 3: Checking DAG...")
    is_dag, cycles = check_dag(kg_structure)
    
    if not is_dag:
        logger.warning(f"Found {len(cycles)} cycles, fixing...")
        for cycle in cycles[:3]:
            logger.warning(f"  Cycle: {' -> '.join(map(str, cycle))}")
        
        kg_structure, removed_count = remove_cycles(kg_structure)
        logger.info(f"Removed {removed_count} edges")
        is_dag, cycles_after = check_dag(kg_structure)
        if not is_dag:
            logger.error(f"Still {len(cycles_after)} cycles after fix!")
            raise RuntimeError("Could not remove all cycles")
        else:
            logger.info("KG fixed, now valid DAG")
    else:
        logger.info("KG is valid DAG (no cycles)")
    
    logger.info(f"Saving KG to {kg_path}...")
    with open(kg_path, 'w', encoding='utf-8') as f:
        json.dump(kg_structure, f, ensure_ascii=False, indent=2)
    total_edges = sum(len(prereqs) for prereqs in kg_structure.values())
    concepts_with_prereqs = sum(1 for prereqs in kg_structure.values() if prereqs)
    
    logger.info(f"KG done: concepts={len(kg_structure)}, edges={total_edges}, with_prereqs={concepts_with_prereqs}")
    
    return kg_structure


def generate_visualization(domain: str, logger: AppLogger):
    """Generate KG visualization (saved under dataset dir)."""
    from scripts.visualize_kg_simple import generate_dot_file, generate_html_visualization
    
    config = load_config(domain=domain)
    data_config = config['data']
    base_path = Path(data_config['base_path']) / domain
    
    kg_path = base_path / data_config['kg_structure_file']
    concept2id_path = base_path / data_config['concept2id_file']
    output_dir = base_path
    logger.info(f"Visualization: {output_dir}")
    generate_dot_file(
        str(kg_path),
        str(concept2id_path),
        str(output_dir / "kg_visualization.dot"),
        max_nodes=150
    )
    generate_html_visualization(
        str(kg_path),
        str(concept2id_path),
        str(output_dir / "kg_visualization.html"),
        max_nodes=150
    )


def main():
    """Generate KG for all datasets."""
    print("=" * 60)
    print("KG generator (LLM + data-driven + DAG)")
    print("=" * 60)
    logger = AppLogger("KG_Generator")
    results = []
    for domain in DATASETS:
        try:
            config = load_config(domain=domain)
            kg_path = Path(config['data']['base_path']) / domain / "KG_structure.json"
            if kg_path.exists():
                logger.info(f"Removing old KG: {kg_path}")
                kg_path.unlink()
            kg = generate_kg_for_domain(domain, logger)
            generate_visualization(domain, logger)
            total_edges = sum(len(prereqs) for prereqs in kg.values())
            results.append((domain, True, f"{len(kg)} concepts, {total_edges} edges"))
        except Exception as e:
            logger.error(f"[{domain}] Failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((domain, False, str(e)))
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for domain, success, message in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {domain}: {message}")
    all_success = all(success for _, success, _ in results)
    if all_success:
        print("\nKG generation done. Visualization: data/{domain}/kg_visualization.html")
        try:
            import webbrowser
            first_domain = results[0][0]
            config = load_config(domain=first_domain)
            html_path = Path(config['data']['base_path']) / first_domain / "kg_visualization.html"
            if html_path.exists():
                print(f"\nOpening: {html_path}")
                webbrowser.open(f'file://{html_path.resolve()}')
        except Exception as e:
            print(f"\nCould not open browser: {e}")
    else:
        print("\nSome datasets failed. Check errors above.")
        sys.exit(1)


if __name__ == '__main__':
    main()
