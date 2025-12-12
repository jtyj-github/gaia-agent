"""
Generate GAIA Predictions for Submission.
Runs the multi-agent system on GAIA test questions and generates submission file.
"""

import argparse
from datasets import load_dataset
from main import initialize_llm, initialize_agents, load_config
from agents.orchestrator import Orchestrator
from agents.validator import AnswerValidator
from evaluation.metrics import MetricsTracker
from utils import setup_logger
from dotenv import load_dotenv
from tqdm import tqdm

logger = setup_logger("generate_predictions")


def generate_predictions(limit: int = None, split: str = "test", output_file: str = None):
    """
    Generate predictions for GAIA submission.

    Args:
        limit: Maximum number of questions to process (None for all)
        split: Dataset split ('test' for submission, 'validation' for testing)
        output_file: Custom output filename (default: auto-generated timestamp)

    Returns:
        Path to generated submission file
    """
    # Load environment
    load_dotenv()

    logger.info("="*60)
    logger.info(f"GAIA Prediction Generation - {split} split")
    if limit:
        logger.info(f"Processing first {limit} questions")
    logger.info("="*60)

    # Load configs
    logger.info("Loading configurations...")
    model_config = load_config('config/model_config.yaml')
    agent_config = load_config('config/agent_config.yaml')

    # Get GAIA system prompt
    gaia_prompt = agent_config.get('system_prompt', '')

    # Initialize system
    logger.info("Initializing system...")
    llm = initialize_llm(model_config)
    agents = initialize_agents(llm, agent_config)

    # Create orchestrator
    orchestrator = Orchestrator(
        llm=llm,
        agents=agents,
        config=agent_config['orchestrator'],
        gaia_prompt=gaia_prompt
    )

    # Get validator for answer extraction
    validator = AnswerValidator(agent_config['validator'])

    # Load GAIA dataset
    logger.info(f"Loading GAIA {split} dataset...")
    try:
        # GAIA requires a config name - use '2023_all' for all difficulty levels
        dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
        logger.info(f"Loaded {len(dataset)} questions")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
        logger.info(f"Limited to {len(dataset)} questions")

    # Initialize metrics tracker
    tracker = MetricsTracker()

    logger.info(f"Processing {len(dataset)} questions...")
    logger.info("="*60)

    # Process each question
    for item in tqdm(dataset, desc="Generating predictions"):
        task_id = item['task_id']
        question = item['Question']
        file_path = item.get('file_path')  # If file is attached
        level = item.get('Level', 'Unknown')
        ground_truth = item.get('Final answer') if split == 'validation' else None  # Only validation has ground truth

        logger.info(f"\nTask: {task_id} (Level {level})")
        logger.info(f"Question: {question[:100]}...")

        try:
            # Run orchestrator
            result = orchestrator.run(question, file_path)

            # Extract final answer using validator
            full_response = result['answer']
            final_answer = validator.extract_final_answer(full_response)

            # Normalize for GAIA format
            normalized_answer = validator.normalize_for_gaia(final_answer)

            # Log prediction with reasoning trace and ground truth
            tracker.log_prediction(
                task_id=task_id,
                question=question,
                prediction=normalized_answer,
                ground_truth=ground_truth,
                reasoning_trace=result['reasoning_trace'],
                level=level
            )

            logger.info(f"Answer: {normalized_answer}")
            if ground_truth:
                logger.info(f"Ground Truth: {ground_truth}")
            logger.info(f"Steps: {result['steps']}")

        except Exception as e:
            logger.error(f"Error processing task {task_id}: {e}", exc_info=True)
            tracker.log_error(task_id, str(e))
            # Log empty prediction for failed tasks
            tracker.log_prediction(
                task_id=task_id,
                question=question,
                prediction="",
                ground_truth=ground_truth,
                reasoning_trace=[f"Error: {str(e)}"],
                level=level
            )

    logger.info("\n" + "="*60)
    logger.info("Prediction Generation Complete")
    logger.info("="*60)

    # Compute metrics (only for validation split which has ground truth)
    if split == 'validation':
        logger.info("Computing metrics using GAIA scorer...")
        metrics = tracker.compute_metrics()
    else:
        logger.info("Test split - no ground truth available for scoring")
        metrics = {
            'total': len(tracker.current_run['predictions']),
            'error_count': len(tracker.current_run['errors'])
        }
        tracker.set_metrics(metrics)

    # Save submission file
    logger.info("Saving submission file...")
    submission_file = tracker.save_submission_format(filename=output_file)

    # Save detailed results with scores
    if split == 'validation':
        results_file = tracker.save(filename=output_file.replace('.jsonl', '_detailed.json') if output_file else None)
        logger.info(f"Detailed results saved to: {results_file}")

    # Print summary
    total_predictions = len(tracker.current_run['predictions'])
    total_errors = len(tracker.current_run['errors'])

    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Total questions processed: {total_predictions}")
    print(f"Successful predictions: {total_predictions - total_errors}")
    print(f"Failed predictions: {total_errors}")

    # Print accuracy metrics if validation split
    if split == 'validation':
        print("\n" + "="*60)
        print("ACCURACY METRICS (Official GAIA Scorer)")
        print("="*60)
        print(f"Overall Accuracy: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        print(f"\nBy Difficulty Level:")
        for level in [1, 2, 3]:
            acc = metrics.get(f'level_{level}_accuracy', 0)
            correct = metrics.get(f'level_{level}_correct', 0)
            total = metrics.get(f'level_{level}_total', 0)
            print(f"  Level {level}: {acc:.2%} ({correct}/{total})")

    print("\n" + "="*60)
    print("FILES GENERATED")
    print("="*60)
    print(f"Submission file: {submission_file}")
    if split == 'validation':
        print(f"Detailed results: {results_file}")
    print("="*60)

    if total_errors > 0:
        print("\n⚠ Warning: Some predictions failed. Check logs for details.")
        print("Failed tasks will have empty answers in submission file.")

    if split == 'test':
        print("\nNext steps for TEST split:")
        print(f"1. Review the submission file: {submission_file}")
        print("2. Upload it to HuggingFace GAIA benchmark submission form")
        print("3. Wait for evaluation results!")
    else:
        print("\nNext steps for VALIDATION split:")
        print(f"1. Review detailed results: {results_file}")
        print(f"2. Analyze errors and improve your system")
        print(f"3. Run on test split when ready: python generate_predictions.py --split test")

    print("="*60 + "\n")

    return submission_file


def main():
    """Command-line interface for prediction generation."""
    parser = argparse.ArgumentParser(
        description="Generate predictions for GAIA benchmark submission"
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of questions to process (default: all)'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['validation', 'test'],
        help='Dataset split (default: test for submission, validation for testing)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output filename (default: auto-generated with timestamp)'
    )

    args = parser.parse_args()

    # Generate predictions
    submission_file = generate_predictions(
        limit=args.limit,
        split=args.split,
        output_file=args.output
    )

    print(f"\n✓ Submission file ready: {submission_file}")


if __name__ == "__main__":
    main()
