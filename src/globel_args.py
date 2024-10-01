import torch
import numpy as np
import os

def set_seed(seed):  # Define a function to set the random seed for reproducibility.
    torch.manual_seed(seed)  # Set the seed for generating random numbers in PyTorch (CPU).
    # random.seed(seed)  # (Commented out) Set the seed for Python's built-in random number generator.
    np.random.seed(seed)  # Set the seed for NumPy's random number generator.

    os.environ['PYTHONHASHSEED'] = str(seed)  # Set the hash seed for Python, which can affect random operations.

    if torch.cuda.is_available():  # Check if a GPU is available.
        torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs if available.
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN.
        torch.backends.cudnn.benchmark = False  # Disable benchmarking to maintain determinism.

# Set the device to CPU for tensor computations
device = torch.device('cpu')  # Specify the device (CPU) for tensor operations.

