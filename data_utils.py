import os
from dotenv import load_dotenv

load_dotenv()

DATASET_BASE_PATH = os.getenv("DATASET_BASE_PATH", "c:/study/Robotics/libero_dataset/IPEC-COMMUNITY")

# Sequence of datasets for continual learning
TASK_SEQUENCE = ['spatial', 'goal', 'object', '10']

# Mapping from sequence name to exact directory name
DATASET_DIR_MAPPING = {
    'spatial': 'libero_spatial_no_noops_1.0.0_lerobot',
    'goal': 'libero_goal_no_noops_1.0.0_lerobot',
    'object': 'libero_object_no_noops_1.0.0_lerobot',
    '10': 'libero_10_no_noops_1.0.0_lerobot'
}

def get_dataset_path(task_name: str) -> str:
    """Returns the absolute path to the dataset directory for a given task name."""
    if task_name not in DATASET_DIR_MAPPING:
        raise ValueError(f"Task {task_name} is not defined in DATASET_DIR_MAPPING.")
    return os.path.join(DATASET_BASE_PATH, DATASET_DIR_MAPPING[task_name])
