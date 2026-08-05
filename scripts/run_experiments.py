from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_eval import run_evaluation


DATASET_CONFIGS = {
    "Logistics": "config/config_logistics.yaml",
    "Mechanical_Physics": "config/config_physics.yaml",
    "PHP": "config/config_php.yaml",
}

VARIANT_CONFIGS = {
    "full": "full",
    "wo_cb": "wo_cb",
    "wo_eg": "wo_eg",
    "wo_cbeg": "wo_cbeg",
}


def run_experiments(
    datasets: List[str],
    steps: List[int],
    variants: List[str],
    mock_llm: bool = False,
    max_students: int = None,
) -> Dict:
    results = {}
    for dataset in datasets:
        config_path = DATASET_CONFIGS[dataset]
        results[dataset] = {}
        for variant in variants:
            results[dataset][variant] = {}
            for step in steps:
                summary = run_evaluation(
                    config_path=config_path,
                    max_students=max_students,
                    max_steps=step,
                    variant=variant,
                    mock_llm=mock_llm,
                    verbose=False,
                )
                results[dataset][variant][str(step)] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CBEGRec experiments")
    parser.add_argument("--datasets", nargs="+", default=["Logistics", "Mechanical_Physics", "PHP"])
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--variants", nargs="+", default=["full", "wo_cb", "wo_eg", "wo_cbeg"])
    parser.add_argument("--max_students", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/experiments_summary.json")
    parser.add_argument("--mock_llm", action="store_true")
    args = parser.parse_args()

    results = run_experiments(args.datasets, args.steps, args.variants, mock_llm=args.mock_llm, max_students=args.max_students)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved experiment summary to {output_path}")


if __name__ == "__main__":
    main()
