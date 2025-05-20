#!/bin/bash

MAX_ATTEMPTS=10
attempt=1
success=false

while [ $attempt -le $MAX_ATTEMPTS ] && [ "$success" = false ]
do
    echo "Attempt $attempt of $MAX_ATTEMPTS"
    mpirun -n 4 singularity exec --nv hierdamap.sif python3 train_our_mapping_b2s.py True
    
    if [ $? -eq 0 ]; then
        success=true
        echo "Command completed successfully"
    else
        echo "Command failed, retrying..."
        attempt=$((attempt+1))
        sleep 2
    fi
done

if [ "$success" = false ]; then
    echo "Failed after $MAX_ATTEMPTS attempts"
    exit 1
fi
