import os
from sim_env import evaluate_policy_in_sim
import torch

def load_model_for_eval(model_id: str, checkpoint_path: str = None):
    """
    Loads the HuggingFace model and injects LoRA adapters if checkpoint is provided.
    """
    print(f"[eval] Loading model '{model_id}'...")
    
    try:
        # Attempt to use the generic PreTrainedPolicy loader
        from lerobot.policies.pretrained import PreTrainedPolicy
        
        # Load the base policy
        policy = PreTrainedPolicy.from_pretrained(model_id)
        
        if checkpoint_path:
            print(f"[eval] Injecting LoRA weights from '{checkpoint_path}'...")
            from peft import PeftModel
            policy = PeftModel.from_pretrained(policy, checkpoint_path)
            
        policy.eval()
        return policy
    except Exception as e:
        print(f"[eval] Warning: LeRobot/PEFT not fully installed. Falling back to Mock Model. ({e})")
        
        class MockPolicy:
            def select_action(self, obs):
                return torch.zeros(7) # Mock 7D action
                
            def eval(self):
                pass
                
        return MockPolicy()

def run_evaluation(model, dataset_path: str, num_episodes: int = 2) -> float:
    """
    Runs evaluation of the given model on the specific dataset's 2 test tasks.
    Returns the success rate.
    """
    success_rate = evaluate_policy_in_sim(model, dataset_path, num_episodes)
    return success_rate
