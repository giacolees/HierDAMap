# Create a test.py file
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
torch._C._cuda_init()
