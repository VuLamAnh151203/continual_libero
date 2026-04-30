import os
import json
import numpy as np
from dotenv import load_dotenv

from continual_libero.data_utils import TASK_SEQUENCE, get_dataset_path
from continual_libero.train import fine_tune_lora
from continual_libero.pipeline1_eval import evaluate_sequential
from continual_libero.metrics import summarize_metrics

load_dotenv()

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "lerobot/xvla-base")
BASELINE_MODEL_ID = os.getenv("BASELINE_MODEL_ID", "lerobot/xvla-libero")
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "c:/study/Robotics/continual_libero")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    num_tasks = len(TASK_SEQUENCE)
    
    # accuracy_matrix[i, j] will store performance on task j after training on task i
    accuracy_matrix = np.full((num_tasks, num_tasks), np.nan)
    
    # Step 0: Zero-Shot Baseline Evaluation
    print("==================================================")
    print("STEP 0: Zero-Shot Baseline Evaluation")
    print(f"Model: {BASELINE_MODEL_ID}")
    print("==================================================")
    baseline_results = evaluate_sequential(
        model_id=BASELINE_MODEL_ID, 
        task_sequence=TASK_SEQUENCE, 
        checkpoint_path=None
    )
    
    with open(os.path.join(RESULTS_DIR, "baseline_results.json"), "w") as f:
        json.dump(baseline_results, f, indent=4)
        
    # Step 1: Sequential LoRA Fine-Tuning
    current_checkpoint = None
    
    for i, task_name in enumerate(TASK_SEQUENCE):
        print("\n==================================================")
        print(f"STEP 1: Sequential LoRA Fine-Tuning - Phase {i+1}/{num_tasks}")
        print(f"Current Training Task: {task_name}")
        print("==================================================")
        
        dataset_path = get_dataset_path(task_name)
        
        # Train on the current task
        # Note: If current_checkpoint is not None, we would load it to continue training
        # For simplicity, fine_tune_lora is assumed to handle the base_model + previous adapters
        current_checkpoint = fine_tune_lora(
            model_id=BASE_MODEL_ID,
            dataset_path=dataset_path,
            output_dir=CHECKPOINT_DIR,
            num_epochs=1 # Dry-run config
        )
        
        # Evaluate on all tasks learned SO FAR (j <= i)
        tasks_to_evaluate = TASK_SEQUENCE[:i+1]
        eval_results = evaluate_sequential(
            model_id=BASE_MODEL_ID,
            task_sequence=tasks_to_evaluate,
            checkpoint_path=current_checkpoint,
            num_episodes=2 # Dry-run config
        )
        
        # Populate the accuracy matrix
        for j, eval_task in enumerate(tasks_to_evaluate):
            accuracy_matrix[i, j] = eval_results[eval_task]
            
        # Calculate metrics
        metrics_summary = summarize_metrics(accuracy_matrix, TASK_SEQUENCE)
        
        # Save metrics to results
        with open(os.path.join(RESULTS_DIR, f"continual_learning_results_step_{i}.json"), "w") as f:
            json.dump({
                "phase": task_name,
                "metrics": metrics_summary
            }, f, indent=4)
            
    print("\n==================================================")
    print("Continual Learning Pipeline Complete!")
    print("==================================================")
    print("Final Accuracy Matrix (lower triangular):")
    print(accuracy_matrix)

if __name__ == "__main__":
    main()
