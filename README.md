# Navier-Stokes Symphony v13.5
Solver for 3D Navier-Stokes blow-up simulations.
## Dependencies
- `pip install numpy cupy-cuda12x mpi4py h5py torch transformers gym stable-baselines3 pysr`
- Install Lean via `elan`
## Usage
- Run on a SLURM cluster: `sbatch submit_ns_job.sh`
## Sample Data
Sample data for \( N = 4 \) is in `ns_data_*.txt` files, representing initial and blow-up states. Full \( N = 10^5 \) data is summarized in `results_summary.txt`.
