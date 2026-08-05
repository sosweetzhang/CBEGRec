import json
import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.kg.hybrid_kg_construction import HybridKGConstruction, is_dag

DATASETS = ["PHP", "Mechanical_Physics", "Logistics"]


def load_concept2id(path: Path) -> Dict[str, int]:
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
    return {str(v): k for k, v in concept2id.items()}


def prepare_dataset(domain: str, base_path: Path) -> Tuple[bool, str]:
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
            print(f"  Building paper-aligned KG from {records_file.name}...")
            builder = HybridKGConstruction(
                {"kg_construction": {"llm_extraction_enabled": False}},
                llm_client=None,
            )
            kg = builder.construct_hybrid_kg(
                text_data=[],
                records_path=str(records_file),
                problem_info_path=str(data_path / 'problem_info.json'),
                concept2id_path=str(concept2id_path),
            )
            total_edges = sum(len(prereqs) for prereqs in kg.values())
            print(f"  [Generated] KG_structure.json ({num_concepts} concepts, {total_edges} edges, DAG={is_dag(kg)})")
        
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
