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


def run_llm_effects(
    dataset: str,
    models: List[str],
    steps: List[int],
    mock_llm: bool = False,
    max_students: int = None,
) -> Dict:
    config_path = DATASET_CONFIGS[dataset]
    results = {}
    for model in models:
        results[model] = {}
        for step in steps:
            summary = run_evaluation(
                config_path=config_path,
                max_students=max_students,
                max_steps=step,
                variant="full",
                mock_llm=mock_llm,
                config_overrides={"llm": {"model": model}},
                verbose=False,
            )
            results[model][str(step)] = summary
    return results


def main():
    parser = argparse.ArgumentParser(description="Run CBEGRec LLM backbone comparison")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), default="Logistics")
    parser.add_argument("--models", nargs="+", default=["qwen-plus", "gpt-4", "claude-3"])
    parser.add_argument("--steps", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--max_students", type=int, default=None)
    parser.add_argument("--output", type=str, default="outputs/llm_effects_summary.json")
    parser.add_argument("--mock_llm", action="store_true")
    args = parser.parse_args()

    results = run_llm_effects(args.dataset, args.models, args.steps, mock_llm=args.mock_llm, max_students=args.max_students)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved LLM comparison summary to {output_path}")


if __name__ == "__main__":
    main()
