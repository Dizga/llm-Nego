import logging
import torch

# Set up logging configuration
logging.basicConfig(level=logging.INFO)

def log_gpu_usage():
    for i in range(torch.cuda.device_count()):
        gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # Convert bytes to GB
        gpu_memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)    # Convert bytes to GB
        logging.info(f"GPU {i}: Memory Allocated: {gpu_memory_allocated:.2f} GB, Memory Reserved: {gpu_memory_reserved:.2f} GB")