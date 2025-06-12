import numpy as np
import torch

def check_install():
    print("NumPy version:", np.__version__)
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("MPS (Metal) available:", torch.backends.mps.is_available())
    print("NumPy array:", np.array([1, 2, 3]))
    print("PyTorch tensor:", torch.tensor([1, 2, 3]))

if __name__ == "__main__":
    check_install()
