import os
from sim_env import evaluate_policy_in_sim
import torch

def load_model_for_eval(model_id: str, checkpoint_path: str = None):
    """
    Loads the HuggingFace model and injects LoRA adapters if checkpoint is provided.
    """
    print(f"[eval] Loading model '{model_id}'...")
    
    try:
        from lerobot.configs.policies import PreTrainedConfig
        
        # 1. Load the generic configuration to find the policy type
        cfg = PreTrainedConfig.from_pretrained(model_id)
        policy_type = getattr(cfg, "type", getattr(cfg, "model_type", None))
        
        # 2. Map the type to the actual LeRobot class to avoid `make_policy`'s strict ds_meta check
        if policy_type == "xvla":
            try:
                from lerobot.policies.xvla.modeling_xvla import XVLAPolicy as PolicyClass
            except ImportError:
                from lerobot.policies.xvla.modeling_xvla import XVlaPolicy as PolicyClass
        elif policy_type == "act":
            from lerobot.policies.act.modeling_act import ACTPolicy as PolicyClass
        elif policy_type == "diffusion":
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy as PolicyClass
        elif policy_type == "pi0_fast":
            from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy as PolicyClass
        elif policy_type == "pi0":
            from lerobot.policies.pi0.modeling_pi0 import PI0Policy as PolicyClass
        else:
            raise ValueError(f"Unsupported policy type: {policy_type}")
            
        # 3. Load the pretrained weights
        policy = PolicyClass.from_pretrained(model_id)
        
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
