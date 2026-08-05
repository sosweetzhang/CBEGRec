from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.main import Pipeline


DATASET_CONFIGS = {
    "Logistics": "config/config_logistics.yaml",
    "Mechanical_Physics": "config/config_physics.yaml",
    "PHP": "config/config_php.yaml",
}


def export_cases(dataset: str, num_cases: int, output_dir: str, mock_llm: bool = True) -> List[Dict]:
    config_path = DATASET_CONFIGS[dataset]
    pipeline = Pipeline(config_path, config_overrides={"llm": {"mock": mock_llm}})
    records = pipeline.data_loader.load_records()
    cases = []
    for i, (seq_len, question_ids, answers) in enumerate(records):
        if len(cases) >= num_cases:
            break
        initial_len = max(1, int(0.6 * seq_len))
        initial_qids = question_ids[:initial_len]
        initial_answers = answers[:initial_len]
        initial_exercises = pipeline.data_loader.get_problem_texts(initial_qids)
        selected = pipeline.select_target_concept(seq_len, question_ids, initial_exercises, initial_answers)
        if selected is None:
            continue
        target_concept_id, target_concept_name = selected
        path = pipeline.recommend_path(
            student_id=f"student_{i}",
            initial_exercises=initial_exercises,
            initial_answers=initial_answers,
            target_concept=target_concept_id,
            max_steps=3,
            initial_question_ids=initial_qids,
            variant="full",
        )
        if not path:
            continue
        cases.append(
            {
                "student_id": f"student_{i}",
                "target_concept": target_concept_name,
                "target_concept_id": target_concept_id,
                "path": path,
            }
        )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / f"{dataset}_human_eval_cases.json", "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    return cases


def main():
    parser = argparse.ArgumentParser(description="Export cases for human evaluation")
    parser.add_argument("--datasets", nargs="+", default=["Logistics", "Mechanical_Physics", "PHP"])
    parser.add_argument("--num_cases", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs/human_eval")
    args = parser.parse_args()

    for dataset in args.datasets:
        cases = export_cases(dataset, args.num_cases, args.output_dir)
        print(f"Exported {len(cases)} cases for {dataset}")


if __name__ == "__main__":
    main()
