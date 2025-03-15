import torch

def get_device() -> torch.device:
    """Get the best available device for PyTorch.
    
    Returns:
        torch.device: The device to use (cuda, mps, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")