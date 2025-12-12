"""
Metrics Tracker for GAIA evaluation.
Tracks predictions, errors, and generates reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from utils import setup_logger

logger = setup_logger("metrics")


class MetricsTracker:
    """Track and log performance metrics."""

    def __init__(self, output_dir: str = "results"):
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_run = {
            'start_time': datetime.now().isoformat(),
            'predictions': {},
            'metrics': {},
            'errors': []
        }

    def log_prediction(self, task_id: str, question: str, prediction: str,
                      ground_truth: str = None, correct: bool = None,
                      reasoning_trace: List[str] = None):
        """
        Log a single prediction.

        Args:
            task_id: Task identifier
            question: Original question
            prediction: Model's prediction
            ground_truth: Ground truth answer (optional)
            correct: Whether prediction is correct (optional)
            reasoning_trace: Steps taken to reach answer (optional)
        """
        self.current_run['predictions'][task_id] = {
            'question': question,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'correct': correct,
            'reasoning_trace': reasoning_trace or [],
            'timestamp': datetime.now().isoformat()
        }

        logger.debug(f"Logged prediction for task {task_id}")

    def log_error(self, task_id: str, error: str):
        """
        Log an error.

        Args:
            task_id: Task identifier where error occurred
            error: Error message
        """
        self.current_run['errors'].append({
            'task_id': task_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })

        logger.warning(f"Logged error for task {task_id}: {error}")

    def set_metrics(self, metrics: Dict[str, Any]):
        """
        Set final metrics.

        Args:
            metrics: Dictionary of metric values
        """
        self.current_run['metrics'] = metrics
        self.current_run['end_time'] = datetime.now().isoformat()
        logger.info("Metrics set")

    def save(self, filename: str = None) -> Path:
        """
        Save results to file.

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_run, f, indent=2)

        logger.info(f"Results saved to {output_file}")
        return output_file

    def save_submission_format(self, filename: str = None) -> Path:
        """
        Save results in GAIA submission format (JSON Lines).

        Format per line:
        {"task_id": "...", "model_answer": "...", "reasoning_trace": "..."}

        Args:
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.jsonl"

        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            for task_id, pred_data in self.current_run['predictions'].items():
                submission_entry = {
                    'task_id': task_id,
                    'model_answer': pred_data['prediction'],
                    'reasoning_trace': '\n'.join(pred_data.get('reasoning_trace', []))
                }
                f.write(json.dumps(submission_entry) + '\n')

        logger.info(f"Submission file saved to {output_file}")
        return output_file

    def print_summary(self):
        """Print summary of results."""
        metrics = self.current_run['metrics']

        print("\n" + "="*60)
        print("GAIA Benchmark Results")
        print("="*60)
        print(f"Overall Accuracy: {metrics.get('accuracy', 0):.2%}")
        print(f"Correct: {metrics.get('correct', 0)}/{metrics.get('total', 0)}")
        print(f"\nBy Level:")
        print(f"  Level 1: {metrics.get('level_1_accuracy', 0):.2%}")
        print(f"  Level 2: {metrics.get('level_2_accuracy', 0):.2%}")
        print(f"  Level 3: {metrics.get('level_3_accuracy', 0):.2%}")
        print(f"\nErrors: {len(self.current_run['errors'])}")

        if self.current_run['errors']:
            print("\nError Summary:")
            error_types = {}
            for error in self.current_run['errors']:
                error_msg = error['error'][:50]
                error_types[error_msg] = error_types.get(error_msg, 0) + 1

            for error_msg, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {count}x: {error_msg}...")

        print("="*60 + "\n")
