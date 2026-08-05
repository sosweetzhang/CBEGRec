from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from src.evaluation.metrics import LPRMetrics, compute_learning_effectiveness
from src.main import Pipeline
from src.utils.logger import AppLogger


def run_evaluation(
    config_path: Optional[str] = None,
    max_students: Optional[int] = None,
    max_steps: Optional[int] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    variant: str = "full",
    mock_llm: bool = False,
    config_overrides: Optional[dict] = None,
):
    config = load_config(config_path)
    if config_overrides:
        from src.main import _deep_update

        config = _deep_update(config, config_overrides)
    if mock_llm:
        config = {**config, "llm": {**config.get("llm", {}), "mock": True}}
    logger = AppLogger("Evaluation", level="INFO" if verbose else "WARNING")
    max_steps = max_steps or config.get("recommendation", {}).get("max_steps", 20)

    logger.info("=" * 60)
    logger.info("CBEGRec evaluation")
    logger.info("=" * 60)

    pipeline_overrides = dict(config_overrides or {})
    pipeline_overrides["llm"] = {
        **pipeline_overrides.get("llm", {}),
        "mock": mock_llm or pipeline_overrides.get("llm", {}).get("mock", False),
    }
    if not verbose:
        pipeline_overrides["logging"] = {**pipeline_overrides.get("logging", {}), "level": "WARNING"}
    pipeline = Pipeline(config_path, config_overrides=pipeline_overrides if pipeline_overrides else None)
    data_loader = pipeline.data_loader
    all_records = data_loader.load_records()
    if max_students:
        all_records = all_records[:max_students]

    metrics = LPRMetrics()
    all_results = []
    student_results = []

    for i, (seq_len, question_ids, answers) in enumerate(all_records):
        student_id = f"student_{i}"
        initial_len = max(1, int(0.6 * seq_len))
        initial_qids = question_ids[:initial_len]
        initial_answers = answers[:initial_len]
        initial_exercises = data_loader.get_problem_texts(initial_qids)
        if not initial_exercises:
            continue
        selected = pipeline.select_target_concept(seq_len, question_ids, initial_exercises, initial_answers)
        if selected is None:
            continue
        target_concept_id, target_concept_name = selected
        path = pipeline.recommend_path(
            student_id=student_id,
            initial_exercises=initial_exercises,
            initial_answers=initial_answers,
            target_concept=target_concept_id,
            max_steps=max_steps,
            initial_question_ids=initial_qids,
            variant=variant,
        )
        if not path:
            continue
        initial_kt_mastery = float(path[0].get("target_mastery_kt_before", 0.0))
        final_kt_mastery = float(path[-1].get("target_mastery_kt", 0.0))
        goal_completed = final_kt_mastery >= float(config.get("ecge", {}).get("mastery_threshold", 0.7))
        metrics.add_episode(
            steps=len(path),
            goal_completed=goal_completed,
            initial_score=initial_kt_mastery,
            final_score=final_kt_mastery,
            max_steps=max_steps,
        )
        ep = compute_learning_effectiveness(initial_kt_mastery, final_kt_mastery)
        student_results.append(
            {
                "student_id": student_id,
                "target_concept": target_concept_name,
                "target_concept_id": target_concept_id,
                "path_length": len(path),
                "goal_completed": goal_completed,
                "Ep": ep,
                "path": path,
                "variant": variant,
            }
        )
        all_results.append(
            {
                "student_id": student_id,
                "target_concept": target_concept_name,
                "path_length": len(path),
                "goal_completed": goal_completed,
                "Ep": ep,
            }
        )

    computed = metrics.compute_all_metrics()
    summary = {
        "variant": variant,
        "total_students": len(all_records),
        "recommended_students": len(student_results),
        "avg_astg": float(computed["ASTG"]) if student_results else 0.0,
        "avg_gcr": float(computed["GCR"]) if student_results else 0.0,
        "avg_ep": float(computed["Ep"]) if student_results else 0.0,
        "metrics": computed,
    }

    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = pipeline.output_manager.run_dir
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(output_path / "evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "student_results": all_results}, f, ensure_ascii=False, indent=2)
    with open(output_path / "full_paths.json", "w", encoding="utf-8") as f:
        json.dump(student_results, f, ensure_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run CBEGRec evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max_students", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--variant", type=str, default="full")
    parser.add_argument("--mock_llm", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    summary = run_evaluation(
        config_path=args.config,
        max_students=args.max_students,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        variant=args.variant,
        mock_llm=args.mock_llm,
    )
    print("\n[OK] Evaluation complete.")
    print(f"  Variant: {summary['variant']}")
    print(f"  Recommended: {summary['recommended_students']}/{summary['total_students']} students")
    print(f"  ASTG: {summary['avg_astg']:.2f}")
    print(f"  GCR: {summary['avg_gcr']:.4f}")
    print(f"  Ep: {summary['avg_ep']:.4f}")


if __name__ == "__main__":
    main()
