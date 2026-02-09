"""
CBEGRec evaluation script: run full learning path recommendation and evaluation.
"""
import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from config import load_config
from src.main import Pipeline
from src.utils.logger import AppLogger
from src.evaluation.metrics import LPRMetrics
from src.data.data_loader import RecordsDataLoader

def run_evaluation(
    config_path: str = None,
    max_students: int = None,
    max_steps: int = None,
    output_dir: str = None,
    verbose: bool = True
):
    """
    Run full CBEGRec evaluation.

    Args:
        config_path: Path to config file.
        max_students: Max number of students (None = all).
        max_steps: Max recommendation steps.
        output_dir: Output directory.
        verbose: Whether to show detailed logs.
    """
    config = load_config(config_path)
    logger = AppLogger("Evaluation", level='INFO' if verbose else 'WARNING')
    
    rec_config = config.get('recommendation', {})
    max_steps = max_steps or rec_config.get('max_steps', 50)
    data_config = config['data']
    
    logger.info("=" * 60)
    logger.info("CBEGRec evaluation")
    logger.info("=" * 60)

    try:
        pipeline = Pipeline(config_path)
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    logger.info("Loading student records...")
    data_loader = pipeline.data_loader
    all_records = data_loader.load_records()
    
    total = len(all_records)
    if max_students:
        all_records = all_records[:max_students]
        logger.info(f"Using {len(all_records)} students (limited by max_students)")
    else:
        eval_config = config.get('evaluation', {})
        student_ratio = float(eval_config.get('student_ratio', 1.0))
        student_ratio = max(0.0, min(1.0, student_ratio))
        if student_ratio < 1.0:
            n_use = max(1, int(total * student_ratio))
            all_records = all_records[:n_use]
            logger.info(f"Using {len(all_records)} students ({student_ratio:.0%} of {total})")
        else:
            logger.info(f"Using all {len(all_records)} students")
    
    metrics = LPRMetrics()
    all_results = []
    student_results = []
    recommended_students = 0
    not_recommended_students = 0
    goal_completed_students = 0
    goal_not_completed_students = 0
    
    for i, (seq_len, question_ids, answers) in enumerate(all_records):
        student_id = f"student_{i}"
        
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Processing Student {i+1}/{len(all_records)}: {student_id}")
        logger.info("=" * 70)
        
        if verbose and (i + 1) % 10 == 0:
            logger.info(f"Progress: {i+1}/{len(all_records)} students processed...")
        
        try:
            initial_len = max(1, int(0.6 * seq_len))
            initial_qids = question_ids[:initial_len]
            initial_answers = answers[:initial_len]
            initial_exercises = data_loader.get_problem_texts(initial_qids)
            
            if not initial_exercises:
                if verbose:
                    logger.warning(f"Student {student_id}: No exercise texts, skipping")
                not_recommended_students += 1
                continue
            selected = pipeline.select_target_concept(
                seq_len, question_ids, initial_exercises, initial_answers
            )
            if selected is not None:
                target_concept_id, target_concept_name = selected
                logger.info(f"Target Concept: {target_concept_name} (ID: {target_concept_id})")
                target_start_idx = int(0.8 * seq_len)
                path = pipeline.recommend_path(
                    student_id=student_id,
                    initial_exercises=initial_exercises,
                    initial_answers=initial_answers,
                    target_concept=target_concept_id,
                    max_steps=max_steps,
                    initial_question_ids=initial_qids
                )
                mastery_threshold = float(config.get('ecge', {}).get('mastery_threshold', 0.7))
                is_skipped = (path and len(path) == 1 and path[0].get('skipped', False))
                if is_skipped:
                    logger.info("Recommendation skipped: target already mastered (KT)")
                    logger.info(f"  Initial KT mastery: {path[0].get('initial_kt_mastery', 0):.4f}")
                    logger.info(f"Result: target already mastered, excluded from metrics")
                    logger.info("-" * 70)
                    not_recommended_students += 1
                    continue
                
                if path:
                    initial_kt_mastery = float(path[0].get('target_mastery_kt_before', 
                                                           path[0].get('target_mastery_kt', 0.0)))
                    final_kt_mastery = float(path[-1].get('target_mastery_kt', 0.0))
                else:
                    initial_kt_mastery = 0.0
                    final_kt_mastery = 0.0
                goal_completed = final_kt_mastery >= mastery_threshold
                
                actual_steps = len(path)
                astg = actual_steps if goal_completed else (max_steps + 1)
                gcr = 1.0 if goal_completed else 0.0
                from src.evaluation.metrics import compute_learning_effectiveness
                learning_effectiveness = compute_learning_effectiveness(initial_kt_mastery, final_kt_mastery)
                metrics.add_episode(
                    steps=actual_steps,
                    goal_completed=goal_completed,
                    initial_score=initial_kt_mastery,
                    final_score=final_kt_mastery,
                    max_steps=max_steps
                )
                
                true_path_length = target_start_idx - initial_len
                
                result = {
                    'student_id': student_id,
                    'target_concept': target_concept_name,
                    'target_concept_id': target_concept_id,
                    'path_length': actual_steps,
                    'true_path_length': true_path_length,
                    'astg': astg,
                    'gcr': gcr,
                    'Ep': learning_effectiveness,
                    'goal_completed': goal_completed,
                    'initial_kt_mastery': initial_kt_mastery,
                    'final_kt_mastery': final_kt_mastery,
                    'mastery_threshold': mastery_threshold,
                    'path': path
                }
                
                student_results.append(result)
                all_results.append({
                    'student_id': student_id,
                    'target_concept': target_concept_name,
                    'path_length': actual_steps,
                    'astg': astg,
                    'gcr': gcr,
                    'Ep': learning_effectiveness
                })
                
                recommended_students += 1
                if goal_completed:
                    goal_completed_students += 1
                else:
                    goal_not_completed_students += 1
                
                status = "goal completed" if goal_completed else "goal not completed"
                logger.info(f"Result: {status}, steps={actual_steps}, ASTG={astg:.2f}, "
                          f"Ep={learning_effectiveness:.4f}")
                logger.info("-" * 70)
            else:
                logger.warning(
                    f"No target concept selected: "
                    f"sequence too short, or all concepts in 80-100% range are already mastered"
                )
                logger.info("-" * 70)
                not_recommended_students += 1
                
        except Exception as e:
            logger.error(f"Error processing student {student_id}: {e}")
            logger.info("-" * 70)
            if verbose:
                import traceback
                traceback.print_exc()
            not_recommended_students += 1
            continue
    if student_results:
        computed = metrics.compute_all_metrics()
        avg_astg = computed['ASTG']
        avg_gcr = computed['GCR']
        avg_ep = computed['Ep']
        recommendation_rate = recommended_students / len(all_records)
        goal_completion_rate = goal_completed_students / recommended_students if recommended_students > 0 else 0.0
        
        summary = {
            'total_students': len(all_records),
            'recommended_students': recommended_students,
            'not_recommended_students': not_recommended_students,
            'goal_completed_students': goal_completed_students,
            'goal_not_completed_students': goal_not_completed_students,
            'recommendation_rate': recommendation_rate,
            'goal_completion_rate': goal_completion_rate,
            'avg_astg': float(avg_astg),
            'avg_gcr': float(avg_gcr),
            'avg_ep': float(avg_ep),
            'metrics': computed
        }
    else:
        logger.error("No successful recommendations!")
        summary = {
            'total_students': len(all_records),
            'recommended_students': 0,
            'not_recommended_students': len(all_records),
            'goal_completed_students': 0,
            'goal_not_completed_students': 0,
            'recommendation_rate': 0.0,
            'goal_completion_rate': 0.0,
            'error': 'No successful recommendations'
        }
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation summary")
    logger.info("=" * 60)
    logger.info(f"Total students: {summary['total_students']}")
    logger.info(f"Recommended: {summary['recommended_students']} (in metrics)")
    logger.info(f"Not recommended: {summary['not_recommended_students']} (target already mastered)")
    
    if 'recommended_students' in summary and summary['recommended_students'] > 0:
        logger.info(f"  Goal completed: {summary['goal_completed_students']}")
        logger.info(f"  Goal not completed: {summary['goal_not_completed_students']}")
        logger.info(f"Goal completion rate: {summary['goal_completion_rate']:.2%}")
        
        if 'avg_astg' in summary:
            logger.info(f"\nMetrics (recommended students only):")
            logger.info(f"  ASTG: {summary['avg_astg']:.2f}")
            logger.info(f"  GCR: {summary['avg_gcr']:.2%}")
            logger.info(f"  Ep: {summary['avg_ep']:.4f} ({summary['avg_ep']*100:.2f}%)")
    
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = pipeline.output_manager.run_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_path / "evaluation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"\nSummary saved: {summary_path}")
    
    detailed_path = output_path / "evaluation_details.json"
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'student_results': all_results
        }, f, indent=2, ensure_ascii=False)
    logger.info(f"Details saved: {detailed_path}")
    
    if verbose:
        full_path = output_path / "full_paths.json"
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(student_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Full paths saved: {full_path}")
    
    logger.info("=" * 60)
    logger.info("Evaluation complete.")
    logger.info("=" * 60)
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Run CBEGRec evaluation')
    parser.add_argument('--config', type=str, default=None,
                       help='Config path (default: config/config.yaml)')
    parser.add_argument('--max_students', type=int, default=None,
                       help='Max students (default: all)')
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Max recommendation steps (default: from config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: outputs/{run_id})')
    parser.add_argument('--quiet', action='store_true',
                       help='Quiet mode')
    
    args = parser.parse_args()
    
    try:
        summary = run_evaluation(
            config_path=args.config,
            max_students=args.max_students,
            max_steps=args.max_steps,
            output_dir=args.output_dir,
            verbose=not args.quiet
        )
        
        if summary:
            print("\n[OK] Evaluation complete.")
            print(f"  Recommended: {summary['recommended_students']}/{summary['total_students']} students")
            print(f"  Not recommended: {summary['not_recommended_students']} (target already mastered)")
            if 'recommended_students' in summary and summary['recommended_students'] > 0:
                print(f"  Goal completion rate: {summary['goal_completion_rate']:.2%}")
                if 'avg_astg' in summary:
                    print(f"  ASTG: {summary['avg_astg']:.2f}")
                    print(f"  Ep: {summary['avg_ep']:.4f} ({summary['avg_ep']*100:.2f}%)")
        else:
            print("\n[FAIL] Evaluation failed.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
