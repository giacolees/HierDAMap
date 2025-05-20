import torch
print('Clearing CUDA cache...')
torch.cuda.empty_cache()
print('CUDA cache cleared successfully')
