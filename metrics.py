import numpy as np
from typing import List, Dict

def calculate_average_performance(accuracy_matrix: np.ndarray, current_task_index: int) -> float:
    """
    Calculate the average performance over all tasks seen so far.
    accuracy_matrix[i, j] = accuracy on task j after training on task i.
    """
    # We take the mean of the current row (up to current_task_index)
    return float(np.mean(accuracy_matrix[current_task_index, :current_task_index + 1]))

def calculate_forgetting(accuracy_matrix: np.ndarray, current_task_index: int) -> float:
    """
    Calculate the forgetting ratio (Backward Transfer) across previously seen tasks.
    Forgetting for task j after training on task i (i > j) is:
    max_{k \in 0..i-1}(accuracy_matrix[k, j]) - accuracy_matrix[i, j]
    Average forgetting is the mean of forgetting for all j < i.
    """
    if current_task_index == 0:
        return 0.0 # No forgetting on the first task
    
    forgetting_list = []
    for j in range(current_task_index):
        # max performance on task j in previous steps
        best_past_performance = np.max(accuracy_matrix[:current_task_index, j])
        current_performance = accuracy_matrix[current_task_index, j]
        forgetting_list.append(best_past_performance - current_performance)
        
    return float(np.mean(forgetting_list))

def summarize_metrics(accuracy_matrix: np.ndarray, task_sequence: List[str]) -> Dict:
    """
    Generate a summary dictionary of Average Performance and Forgetting 
    at each stage of training.
    """
    num_tasks = len(task_sequence)
    summary = {}
    
    for i in range(num_tasks):
        # We only calculate if the row is fully populated up to i
        if not np.isnan(accuracy_matrix[i, 0]):
            avg_perf = calculate_average_performance(accuracy_matrix, i)
            forgetting = calculate_forgetting(accuracy_matrix, i)
            summary[f"After_Train_{task_sequence[i]}"] = {
                "Average_Performance": avg_perf,
                "Forgetting": forgetting,
                "Individual_Accuracies": {task_sequence[j]: float(accuracy_matrix[i, j]) for j in range(i + 1)}
            }
            
    return summary
