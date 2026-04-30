import os
from continual_libero.pipeline1_eval import evaluate_sequential
from continual_libero.data_utils import TASK_SEQUENCE
from dotenv import load_dotenv

load_dotenv()

def test_baseline():
    """
    Standalone script to test the evaluation pipeline with the zero-shot baseline model.
    """
    BASELINE_MODEL_ID = os.getenv("BASELINE_MODEL_ID", "lerobot/xvla-libero")
    
    print("==================================================")
    print("Testing Evaluation Pipeline with X-VLA Baseline")
    print(f"Model ID: {BASELINE_MODEL_ID}")
    print("==================================================")
    
    # To run a quick test, we can evaluate on just the first task suite
    test_sequence = [TASK_SEQUENCE[0]]
    
    print(f"Running evaluation on suite: {test_sequence}")
    results = evaluate_sequential(
        model_id=BASELINE_MODEL_ID,
        task_sequence=test_sequence,
        checkpoint_path=None,
        num_episodes=1  # 1 episode per test task for speed during testing
    )
    
    print("\n==================================================")
    print("Test Results:")
    for task, score in results.items():
        print(f"  {task}: {score:.2f} success rate")
    print("==================================================")
    print("Check the 'evaluation_videos' folder for output videos!")

if __name__ == "__main__":
    test_baseline()
