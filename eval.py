import os
from sim_env import evaluate_policy_in_sim
import torch

def load_model_for_eval(model_id: str, checkpoint_path: str = None):
    """
    Loads a pretrained LeRobot policy for evaluation.
    If checkpoint_path is provided, loads PEFT/LoRA weights on top.
    """
    print(f"[eval] Loading model '{model_id}'...")

    from lerobot.configs import PreTrainedConfig
    from lerobot.policies import get_policy_class

    # 1) Load the pretrained config first
    cfg = PreTrainedConfig.from_pretrained(model_id)

    # 2) Get the correct policy class from the config type
    policy_class = get_policy_class(cfg.type)

    # 3) Load pretrained weights directly (no make_policy call)
    policy = policy_class.from_pretrained(
        pretrained_name_or_path=model_id,
        config=cfg,
        local_files_only=False,  # set True if you only want local files
    )

    # 4) Optionally attach LoRA/PEFT adapter
    if checkpoint_path:
        print(f"[eval] Injecting LoRA weights from '{checkpoint_path}'...")
        from peft import PeftModel
        policy = PeftModel.from_pretrained(policy, checkpoint_path)

    policy.eval()
    return policy
    # except Exception as e:
        # print(f"[eval] Warning: LeRobot/PEFT not fully installed. Falling back to Mock Model. ({e})")
        
        # class MockPolicy:
        #     def select_action(self, obs):
        #         return torch.zeros(7) # Mock 7D action
                
        #     def eval(self):
        #         pass
                
        # return MockPolicy()

def run_evaluation(model, dataset_path: str, num_episodes: int = 2) -> float:
    """
    Runs evaluation of the given model on the specific dataset's 2 test tasks.
    Returns the success rate.
    """
    success_rate = evaluate_policy_in_sim(model, dataset_path, num_episodes)
    return success_rate
