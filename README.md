# Commands to compile and run kdtree.c program

- qsub -q dssc_gpu -l nodes=1:ppn=33 -l walltime=00:10:00 -I
- cd $PBS_O_WORKDIR
- module load  openmpi-4.1.1+gnu-9.3.0
- mpicc -fopenmp kdtree.c -o kdtree.x
- OMP_NUM_THREADS=2 mpirun -np 4 ./kdtree.x | sort 2>/dev/null










