.PHONY: dbuild dup ddown dexec dtest dclean stest stes_mpi

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