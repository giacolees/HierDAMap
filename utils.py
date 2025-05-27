import torch
import torch.distributed as dist
import torch.nn as nn
import socket
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader # Import Dataset
from contextlib import contextmanager
import os
import random
import datetime
import re
import socket
from typing import Optional
import subprocess
import sys


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
def distributed_context(backend='nccl', timeout_seconds=1000, is_distributed=False, gpus_per_node: Optional[int] = None, auto_detect_gpus: bool = True):
    """Context manager for distributed setup and cleanup - supports multi-node."""

    hostname = socket.gethostname()

    rank = -1
    world_size = -1
    local_rank = -1

    if gpus_per_node is None and auto_detect_gpus:
        if torch.cuda.is_available():
            gpus_per_node = torch.cuda.device_count()
            print(f"Auto-detected {gpus_per_node} GPUs on this node")
        else:
            gpus_per_node = 1  # Fallback for CPU-only
            print("No CUDA devices found, using CPU mode")
    elif gpus_per_node is None:
        gpus_per_node = 1  # Default fallback
        print(f"Using default {gpus_per_node} GPUs per node")
    
    
    if 'PMI_RANK' in os.environ:
        rank = int(os.environ.get('PMI_RANK', '0'))
        world_size = int(os.environ.get('PMI_SIZE', '1'))
        
        local_rank = rank % gpus_per_node
        node_id = rank // gpus_per_node 
        total_nodes = (world_size + gpus_per_node - 1) // gpus_per_node  # Ceiling division

        # 🎯 DIAGNOSTIC OUTPUT
        print(f"🖥️  Rank {rank}: Running on hostname '{hostname}'")
        print(f"   Calculated node_id: {node_id}, local_rank: {local_rank}")
        print(f"   Expected: {total_nodes} nodes, {gpus_per_node} GPUs/node")
        
    elif not is_distributed:
        print("Warning: Distributed environment variables not found. Running in single-process mode.")
        rank = 0
        world_size = 1
        local_rank = 0
        node_id = 0
        total_nodes = 1

    else:
        raise RuntimeError("No distributed environment detected. Set PMI_RANK/PMI_SIZE")


    # Set environment variables
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = sys.argv[-1] if len(sys.argv)>2 else 'localhost' 
        
    
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '30006'
        

    # Validate local_rank doesn't exceed available GPUs
    if torch.cuda.is_available() and local_rank >= torch.cuda.device_count():
        print(f"Warning: local_rank {local_rank} >= available GPUs {torch.cuda.device_count()}")
        local_rank = local_rank % torch.cuda.device_count()

    if is_distributed:
        print(f"Process Info: Global Rank={rank}, Local Rank={local_rank}, Node={node_id}/{total_nodes}, "
              f"World Size={world_size}, GPUs/Node={gpus_per_node}")
        

        dist.init_process_group(
            backend=backend, 
            rank=rank, 
            world_size=world_size, 
            timeout=datetime.timedelta(seconds=timeout_seconds)
        )
        
        print(f"Process group initialized: Global Rank={rank}, Local Rank={local_rank}, World Size={world_size}")
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            current_gpu = torch.cuda.current_device()
        else:
            current_gpu = local_rank  # For testing or CPU mode
            
    else:
        if torch.cuda.is_available():
            current_gpu = torch.cuda.current_device()
        else:
            current_gpu = 'CPU'
        print(f"Running in non-distributed mode on device: {current_gpu}")
    
    try:
        # Return comprehensive context information
        yield {
            'rank': rank,                           # Global rank across all processes
            'local_rank': local_rank,               # Local rank on this node (0 to gpus_per_node-1)
            'world_size': world_size,               # Total number of processes
            'device': torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() and local_rank >= 0 else torch.device('cpu'),
            'idx_gpu': local_rank,                  # GPU index on this node
            'gpu': current_gpu,                     # Current GPU device
            'node_id': node_id,                     # Which node this process is on
            'total_nodes': total_nodes,             # Total number of nodes
            'gpus_per_node': gpus_per_node,         # Number of GPUs per node
            'is_master': rank == 0,                 # True if this is the master process
            'is_local_master': local_rank == 0,     # True if this is rank 0 on its node
            'processes_per_node': gpus_per_node,    # Processes per node (usually = gpus_per_node)
        }
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
        print(f"Global Rank {rank} (Local Rank {local_rank}): Distributed resources cleaned up.")

def get_rank():
    try:
        return dist.get_rank()
    except:
        return -1

def save_checkpoint(logdir, epoch, batch_idx, student_model, teacher_model, optimizer, scheduler, 
                   loss_nan_counter, wp, is_distributed=False, checkpoint_name=None):
    """
    Save a checkpoint with all necessary information to resume training.
    
    Args:
        logdir: Directory to save checkpoint
        epoch: Current epoch
        batch_idx: Current batch index within the epoch
        student_model: Student model to save
        teacher_model: Teacher model to save
        optimizer: Optimizer state to save
        scheduler: Learning rate scheduler state to save
        loss_nan_counter: Counter for NaN losses
        wp: Current weight parameter
        is_distributed: Whether using distributed training
        checkpoint_name: Custom checkpoint name (optional)
    """
    if checkpoint_name is None:
        checkpoint_name = f"checkpoint_{epoch}.pt"
    
    checkpoint_path = os.path.join(logdir, checkpoint_name)
    
    # Extract state dicts based on whether using DDP or not
    if is_distributed:
        student_state = student_model.module.state_dict()
        teacher_state = teacher_model.module.state_dict()
    else:
        student_state = student_model.state_dict()
        teacher_state = teacher_model.state_dict()
    
    # Create checkpoint dictionary with all necessary information
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'student_model': student_state,
        'teacher_model': teacher_state,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_nan_counter': loss_nan_counter,
        'wp': wp,
    }
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

def get_latest_model_path(model_dir):
    """
    Get the path to the latest checkpoint and its epoch number.
    
    Args:
        model_dir: Directory containing checkpoints
        
    Returns:
        tuple: (checkpoint_path, epoch_number)
    """
    # Look for checkpoint files
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith('checkpoint_') and f.endswith('.pt')]
    
    if not checkpoint_files:
        return None, 1  # No checkpoints found, start from epoch 1
    
    # Extract epoch numbers
    epoch_numbers = []
    for f in checkpoint_files:
        match = re.search(r'checkpoint_(\d+)(?:_batch_\d+)?\.pt', f)
        if match:
            epoch_numbers.append(int(match.group(1)))
    
    if not epoch_numbers:
        return None, 1  # No valid checkpoint files found
    
    # Find latest epoch
    latest_epoch = max(epoch_numbers)
    
    # Check if there are batch-specific checkpoints for the latest epoch
    batch_checkpoints = [f for f in checkpoint_files if f.startswith(f'checkpoint_{latest_epoch}_batch_')]
    
    if batch_checkpoints:
        # Extract batch numbers
        batch_numbers = [int(re.search(r'_batch_(\d+)\.pt', f).group(1)) for f in batch_checkpoints]
        latest_batch = max(batch_numbers)
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{latest_epoch}_batch_{latest_batch}.pt")
    else:
        # Use the epoch-level checkpoint
        checkpoint_path = os.path.join(model_dir, f"checkpoint_{latest_epoch}.pt")
    
    return checkpoint_path, latest_epoch