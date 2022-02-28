#$ -pe ib 4
mpiexec python3 2d-rbc.py --ra=2081398 --N=128 --max_dt=5e-6 --init_dt=1e-8 --mesh=2,2
