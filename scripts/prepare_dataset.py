"""Prepare datasets: generate id2concept.json, KG_structure.json (data-driven from learning order)."""
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple

sys.path.append(str(Path(__file__).parent.parent))

DATASETS = ["PHP", "Mechanical_Physics", "Logistics"]


def load_concept2id(path: Path) -> Dict[str, int]:
    """Load concept2id (JSON or comma-separated)."""
    concept2id = {}
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if content.startswith('{'):
            concept2id = json.loads(content)
        else:
            for line in content.split('\n'):
                line = line.strip()
                if line and ',' in line:
                    parts = line.rsplit(',', 1)
                    concept = parts[0].strip().strip('"')
                    cid = int(parts[1].strip())
                    concept2id[concept] = cid
    return concept2id


def generate_id2concept(concept2id: Dict[str, int]) -> Dict[str, str]:
    """Build id2concept mapping."""
    return {str(v): k for k, v in concept2id.items()}


def load_records(records_path: Path) -> List[Dict]:
    """
    Load learning records. Supports JSON Lines or N / problem_ids / answers format.
    """
    records = []
    
    with open(records_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    lines = content.split('\n')
    if lines and lines[0].strip().startswith('{'):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError:
                continue
        return records
    i = 0
    student_id = 0
    while i < len(lines):
        try:
            num_problems = int(lines[i].strip())
            i += 1
            if i >= len(lines):
                break
            problem_ids = lines[i].strip().split()
            i += 1
            if i >= len(lines):
                break
            answers = lines[i].strip().split()
            i += 1
            student_id += 1
            for j, (pid, ans) in enumerate(zip(problem_ids, answers)):
                records.append({
                    'user_id': student_id,
                    'exer_id': int(pid),
                    'score': int(ans),
                    'timestamp': j
                })
        except (ValueError, IndexError):
            i += 1
            continue
    
    return records


def load_problem_info(path: Path) -> Dict:
    """Load problem info."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_kg_from_data(
    records: List[Dict],
    problem_info: Dict,
    num_concepts: int,
    min_support: int = 5,
    min_confidence: float = 0.3
) -> Dict[str, List[str]]:
    """Build KG from learning order (A before B and A more correct -> A prereq of B)."""
    problem_to_concept = {}
    for pid, info in problem_info.items():
        if 'concept_routes' in info and info['concept_routes']:
            concept_id = info['concept_routes'][0]
            problem_to_concept[pid] = concept_id
        elif 'concept_id' in info:
            problem_to_concept[pid] = info['concept_id']
    student_sequences = defaultdict(list)
    
    for record in records:
        student_id = record.get('user_id') or record.get('student_id')
        problem_id = str(record.get('exer_id') or record.get('problem_id'))
        correct = record.get('score', 0) or record.get('correct', 0)
        timestamp = record.get('timestamp', record.get('time', len(student_sequences[student_id])))
        
        if problem_id in problem_to_concept:
            concept_id = problem_to_concept[problem_id]
            student_sequences[student_id].append((concept_id, correct, timestamp))
    pair_stats = defaultdict(lambda: {'count': 0, 'a_before_b': 0, 'a_correct_when_before': 0})
    
    for student_id, sequence in student_sequences.items():
        if len(sequence) < 2:
            continue
        sequence.sort(key=lambda x: x[2])
        concept_first_pos = {}
        concept_first_correct = {}
        
        for i, (concept_id, correct, _) in enumerate(sequence):
            if concept_id not in concept_first_pos:
                concept_first_pos[concept_id] = i
                concept_first_correct[concept_id] = correct
        concepts = list(concept_first_pos.keys())
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                if concept_a == concept_b:
                    continue
                
                pos_a = concept_first_pos[concept_a]
                pos_b = concept_first_pos[concept_b]
                
                if pos_a < pos_b:
                    pair_stats[(concept_a, concept_b)]['count'] += 1
                    pair_stats[(concept_a, concept_b)]['a_before_b'] += 1
                    if concept_first_correct[concept_a]:
                        pair_stats[(concept_a, concept_b)]['a_correct_when_before'] += 1
                else:
                    pair_stats[(concept_b, concept_a)]['count'] += 1
                    pair_stats[(concept_b, concept_a)]['a_before_b'] += 1
                    if concept_first_correct[concept_b]:
                        pair_stats[(concept_b, concept_a)]['a_correct_when_before'] += 1
    kg = {str(i): [] for i in range(1, num_concepts + 1)}
    for (concept_a, concept_b), stats in pair_stats.items():
        if stats['count'] < min_support:
            continue
        confidence = stats['a_before_b'] / stats['count'] if stats['count'] > 0 else 0
        if confidence >= min_confidence:
            concept_b_str = str(concept_b)
            concept_a_str = str(concept_a)
            if concept_b_str in kg and concept_a_str not in kg[concept_b_str]:
                kg[concept_b_str].append(concept_a_str)
    kg = remove_cycles(kg)
    
    return kg


def remove_cycles(kg: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Remove cycles from KG to get DAG."""
    graph = defaultdict(set)
    for concept, prereqs in kg.items():
        for prereq in prereqs:
            graph[prereq].add(concept)
    in_degree = defaultdict(int)
    all_nodes = set(kg.keys())
    for concept, prereqs in kg.items():
        for prereq in prereqs:
            in_degree[concept] += 1
            all_nodes.add(prereq)
    queue = [n for n in all_nodes if in_degree[n] == 0]
    sorted_nodes = []
    
    while queue:
        node = queue.pop(0)
        sorted_nodes.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    if len(sorted_nodes) < len(all_nodes):
        cycle_nodes = all_nodes - set(sorted_nodes)
        for node in cycle_nodes:
            if node in kg:
                kg[node] = []
    
    return kg


def prepare_dataset(domain: str, base_path: Path) -> Tuple[bool, str]:
    """
    Prepare one dataset.
    
    Returns:
        (success, message)
    """
    data_path = base_path / domain
    
    print(f"\n{'='*60}")
    print(f"Preparing dataset: {domain}")
    print(f"{'='*60}")
    required_files = ['concept2id.json', 'problem_info.json']
    for fname in required_files:
        if not (data_path / fname).exists():
            return False, f"Missing required file: {fname}"
    concept2id_path = data_path / 'concept2id.json'
    concept2id = load_concept2id(concept2id_path)
    num_concepts = len(concept2id)
    print(f"  Concepts: {num_concepts}")
    id2concept_path = data_path / 'id2concept.json'
    if not id2concept_path.exists():
        id2concept = generate_id2concept(concept2id)
        with open(id2concept_path, 'w', encoding='utf-8') as f:
            json.dump(id2concept, f, ensure_ascii=False, indent=2)
        print(f"  [Generated] id2concept.json ({len(id2concept)} entries)")
    else:
        print(f"  [Exists] id2concept.json")
    kg_path = data_path / 'KG_structure.json'
    if not kg_path.exists():
        records_file = None
        for fname in ['records.txt', 'data_rec.txt']:
            if (data_path / fname).exists():
                records_file = data_path / fname
                break
        if records_file is None:
            kg = {str(i): [] for i in range(1, num_concepts + 1)}
            print(f"  [Warning] No records file, creating empty KG")
        else:
            print(f"  Building KG from {records_file.name}...")
            records = load_records(records_file)
            problem_info = load_problem_info(data_path / 'problem_info.json')
            
            kg = generate_kg_from_data(
                records, 
                problem_info, 
                num_concepts,
                min_support=3,
                min_confidence=0.4
            )
            total_edges = sum(len(prereqs) for prereqs in kg.values())
            print(f"  [Generated] KG_structure.json ({num_concepts} concepts, {total_edges} edges)")
        
        with open(kg_path, 'w', encoding='utf-8') as f:
            json.dump(kg, f, ensure_ascii=False, indent=2)
    else:
        with open(kg_path, 'r', encoding='utf-8') as f:
            kg = json.load(f)
        total_edges = sum(len(prereqs) for prereqs in kg.values())
        print(f"  [Exists] KG_structure.json ({len(kg)} concepts, {total_edges} edges)")
    stats_path = data_path / 'statistic.json'
    if stats_path.exists():
        with open(stats_path, 'r', encoding='utf-8') as f:
            stats = json.load(f)
        print(f"  Stats: {stats.get('student_num', '?')} students, {stats.get('record_num', '?')} records")
    
    return True, "Done"


def main():
    print("=" * 60)
    print("Dataset preparation: generate missing config files per dataset")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent / "data"
    
    results = []
    for domain in DATASETS:
        success, message = prepare_dataset(domain, base_path)
        results.append((domain, success, message))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for domain, success, message in results:
        status = "[OK]" if success else "[FAIL]"
        print(f"  {status} {domain}: {message}")
    print("\n" + "-" * 60)
    print("Dataset info:")
    print("-" * 60)
    print(f"{'Dataset':<20} {'Concepts':<10} {'Students':<10} {'Records':<10}")
    print("-" * 60)
    
    for domain in DATASETS:
        stats_path = base_path / domain / 'statistic.json'
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            concepts = stats.get('concepts_num', '?')
            students = stats.get('student_num', '?')
            records = stats.get('record_num', '?')
            print(f"{domain:<20} {concepts:<10} {students:<10} {records:<10}")
    
    print("-" * 60)
    print("\nAll datasets prepared.")
    print("You can now train models:")
    print("  python scripts/train_ecge_kt_80.py")
    print("  python scripts/train_ecge_50.py")


if __name__ == '__main__':
    main()
