Compile:
        gcc -o sequential_version sequential_version.c
        
    mpicc -O3 -o MPI_parallel_version MPI_parallel_version.c

Run:
    sequential_version <INPUT_FILE>

    mpirun -np <p> MPI_parallel_version <INPUT_FILE>
    
