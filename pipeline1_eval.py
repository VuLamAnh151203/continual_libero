import os
from continual_libero.data_utils import get_dataset_path
from continual_libero.eval import load_model_for_eval, run_evaluation

def evaluate_sequential(model_id: str, task_sequence: list, checkpoint_path: str = None, num_episodes: int = 10) -> dict:
    """
    Evaluates a model sequentially on a given list of task suites.
    Returns a dictionary of success rates for each task.
    """
    print(f"\n--- Starting Pipeline 1: Sequential Evaluation ---")
    model = load_model_for_eval(model_id, checkpoint_path)
    
    results = {}
    for task_name in task_sequence:
        dataset_path = get_dataset_path(task_name)
        success_rate = run_evaluation(model, dataset_path, num_episodes)
        results[task_name] = success_rate
        
    print("--- Pipeline 1 Evaluation Complete ---\n")
    return results
