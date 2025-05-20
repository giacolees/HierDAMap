.PHONY: dbuild dup ddown dexec dtest dclean stest stes_mpi clean-cuda stats-clean-cuda cuda-help

# Default Python interpreter
PYTHON := python

# Path to the script file that empties CUDA cache
EMPTY_CACHE_SCRIPT := empty_cuda_cache.py

NP ?= 4
is_dist ?= False

dbuild:
	docker compose build

dup:
	docker compose up -d

ddown:
	docker compose down

dexec:
	docker compose exec hierdamap bash

dtrain:
	docker compose exec hierdamap python3 train_our_mapping_b2s.py

dtest:
	docker compose exec hierdamap python3 test_hierdamap.py

dclean:
	docker compose down -v

stest:
	singularity exec --nv hierdamap.sif python3 test_hierdamap.py $(is_dist)

stest_mpi:
	mpirun -n $(NP) singularity exec --nv hierdamap.sif python3 test_hierdamap.py $(is_dist)

strain:
	singularity exec --nv hierdamap.sif python3 train_our_mapping_b2s.py $(is_dist)

strain_mpi:
	mpirun -n $(NP) singularity exec --nv hierdamap.sif python3 train_our_mapping_b2s.py $(is_dist)

nvidia-pkill:
	nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Create the empty_cuda_cache.py script if it doesn't exist
$(EMPTY_CACHE_SCRIPT):
	@echo "Creating CUDA cache emptying script..."
	@echo "import torch" > $(EMPTY_CACHE_SCRIPT)
	@echo "print('Clearing CUDA cache...')" >> $(EMPTY_CACHE_SCRIPT)
	@echo "torch.cuda.empty_cache()" >> $(EMPTY_CACHE_SCRIPT)
	@echo "print('CUDA cache cleared successfully')" >> $(EMPTY_CACHE_SCRIPT)
	@chmod +x $(EMPTY_CACHE_SCRIPT)

# Target to empty CUDA cache
clean-cuda: $(EMPTY_CACHE_SCRIPT)
	@echo "Running CUDA cache cleanup..."
	@$(PYTHON) $(EMPTY_CACHE_SCRIPT)

# Show NVIDIA GPU stats before and after clearing
stats-clean-cuda: $(EMPTY_CACHE_SCRIPT)
	@echo "Current GPU memory usage:"
	@nvidia-smi --query-gpu=memory.used --format=csv
	@$(PYTHON) $(EMPTY_CACHE_SCRIPT)
	@echo "GPU memory usage after clearing cache:"
	@nvidia-smi --query-gpu=memory.used --format=csv

# Display help information
cuda-help:
	@echo "CUDA Memory Management Makefile"
	@echo "Available targets:"
	@echo "  clean-cuda       - Clear CUDA cache using PyTorch"
	@echo "  stats-clean-cuda - Show memory stats before and after clearing cache"
	@echo "  help             - Display this help information"

