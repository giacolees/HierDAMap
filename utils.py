import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Import Dataset
from contextlib import contextmanager
import os
import random
import datetime

# Define a simple model
class ToyModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ToyModel, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    def forward(self, x):
        return self.fc(x)

# Define a synthetic dataset
class SyntheticDataset(Dataset):
    def __init__(self, num_samples, input_size, num_classes):
        self.num_samples = num_samples
        self.input_size = input_size
        self.num_classes = num_classes # Ensure this matches target generation
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        # Generate random input data
        # Ensure data type matches what the model expects (e.g., float32)
        data = torch.randn(self.input_size, dtype=torch.float32)
        # Generate random target data
        # For classification, targets are typically long integers (class indices)
        # Ensure target shape is appropriate (e.g., scalar for CrossEntropyLoss)
        target = torch.randint(0, self.num_classes, (1,), dtype=torch.long).squeeze()
        return data, target
        
@contextmanager
def distributed_context(backend='nccl', timeout_seconds=30, is_distributed=False):
    """Context manager for distributed setup and cleanup."""
    rank = -1
    world_size = -1

    if 'PMI_RANK' in os.environ:
        rank = int(os.environ.get('PMI_RANK', '0'))
        world_size = int(os.environ.get('PMI_SIZE', '1'))

    elif not is_distributed: # Fallback for single process or if env vars not set
        print("Warning: Distributed environment variables not found. Running in single-process mode.")
        rank = 0
        world_size = 1

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Set master address and port
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '30002'

    if is_distributed:
        
        # Print debugging information
        print(f"MPI Process: Rank={os.environ['RANK']}, World Size={os.environ['WORLD_SIZE']}, {os.environ['MASTER_ADDR']}, {os.environ['MASTER_PORT']}")

        # Initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=timeout_seconds))
        print(f"Process group initialized: Rank={rank}, World Size={world_size}")
        
        # Set device
        torch.cuda.set_device(rank)
        current_gpu = torch.cuda.current_device()

    else:

        current_gpu = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
        print(f"Running in non-distributed mode on device: {current_gpu}")
    
    try:
        # Return useful values to the context
        yield {
            'rank': rank,
            'world_size': world_size,
            'device': torch.device(f'cuda:{current_gpu}'),
            'idx_gpu': current_gpu,
            'gpu': torch.cuda.current_device()
        }

    finally:

        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: Distributed resources cleaned up.")
        elif not is_distributed:
            if rank == 0:
                print("Non-distributed mode: No distributed resources to clean up.")