.PHONY: test-local test-sbatch

# Local test: run directly with Python (requires singularity)
test-local:
	singularity exec --nv --bind /home/users/caviglia ${HOME}/container-defs/survey.sif \
		python3 test_dvscifar10.py

# HPC test: submit sbatch job (run from repo root)
test-sbatch:
	cd .. && sbatch hpc_tests/test_dvscifar10.sbatch

# Show last output from sbatch test
show-output:
	@ls -lht slurm_outputs/ 2>/dev/null | head -5 || echo "No outputs yet"
	@latest=$$(ls -t slurm_outputs/*.out 2>/dev/null | head -1); \
	if [ -n "$$latest" ]; then tail -50 "$$latest"; else echo "No output files found"; fi
