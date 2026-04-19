import torch
from torch import nn
from pathlib import Path

def save_model(model, target_path, model_name):
    """Saves a trained model to a target path"""
    target_dir_path = Path(target_path)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_file_path = target_dir_path / model_name

    torch.save(model.state_dict(), target_file_path)
    print(f"Model saved to {target_file_path}")

