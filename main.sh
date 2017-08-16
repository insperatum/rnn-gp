#!/bin/bash
if [ "$SLURM_PROCID" -eq "0" ]; then
	echo Args: $@
	module add openmpi/gcc
	module add openmind/miniconda
	source activate ./conda
	PYTHONHASHSEED=0 mpiexec python main.py $@
fi
