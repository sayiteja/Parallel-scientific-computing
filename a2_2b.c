#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<mpi.h>

#define MAX_ITERATIONS 10000
#define TOLERANCE 1e-4
#define PI 3.14159265359

double norm(double **A, double **B, int n){
    double sum = 0.0;
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            sum += (A[i][j] - B[i][j])*(A[i][j] - B[i][j]);
        }
    }
    return sqrt(sum);
}

void jacobi(double del, int n){
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows_per_proc = n/size;
    if (rank == size-1) rows_per_proc += n%size;

    if (rank !=0 && rank != size-1) rows_per_proc += 2;
    else rows_per_proc += 1;

    double** A = (double **)malloc(rows_per_proc * sizeof(double *));
    for(int i = 0; i < rows_per_proc; i++){
        A[i] = (double *)malloc(n * sizeof(double));
    }
    if (rank !=0 && rank != size-1){
        for(int i = 1; i < rows_per_proc-1; i++){
            for (int j = 0; j < n; j++){
                A[i][j] = 0.0;
                if(j==0) A[i][0] = sin(2*PI*((i+rank*(n/size))*del-1));
                if(j==n-1) A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
            }
        }
    }
    else if (rank == 0){
        for(int i = 0; i < rows_per_proc-1; i++){
            for (int j = 0; j < n; j++){
                A[i][j] = 0.0;
                if(j==0) A[i][0] = sin(2*PI*((i+rank*(n/size))*del-1));
                if(j==n-1) A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
            }
        }
    }
    else{
        for(int i = 1; i < rows_per_proc; i++){
            for (int j = 0; j < n; j++){
                A[i][j] = 0.0;
                if(j==0) A[i][0] = sin(2*PI*((i+rank*(n/size))*del-1));
                if(j==n-1) A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
            }
        }
    }

    double** A_old = (double **)malloc(rows_per_proc * sizeof(double *));
    for(int i = 0; i < rows_per_proc; i++){
        A_old[i] = (double *)malloc(n * sizeof(double));
    }

    int iter = 0;
    double x,y,q,error;
    error = 1.0;
    while (iter < MAX_ITERATIONS && error > TOLERANCE){
        if (rank == 0){
            MPI_Send(&A[rows_per_proc-2][0], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[rows_per_proc-1][0], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else if (rank == size-1){
            MPI_Send(&A[1][0], n, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[0][0], n, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else{
            MPI_Send(&A[rows_per_proc-2][0], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[rows_per_proc-1][0], n, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&A[1][0], n, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
            MPI_Recv(&A[0][0], n, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for(int i = 0; i < rows_per_proc; i++){
            for (int j = 0; j < n; j++){
                A_old[i][j] = A[i][j];
            }
        }
        for(int i = 1; i < rows_per_proc-1; i++){
            for (int j = 1; j < n-1; j++){
                x = (i+rank*(n/size))*del - 1;
                y = j*del - 1;
                q = x*x - y*y;
                A[i][j] = 0.25*(A_old[i+1][j]+A_old[i-1][j]+A_old[i][j+1]+A_old[i][j-1]) + 0.25*del*del*q;
            }
        }
        for(int i = 0; i < rows_per_proc; i++){
            A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
        }
        iter++;
        double Norm = norm(A, A_old, rows_per_proc);
        MPI_Allreduce(&Norm, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    if(rank==0) printf("rank: %d, iter: %d, error: %lf\n", rank, iter, error);
    free(A_old);
    free(A);
}

int main(int argc, char** argv){
    double del = 0.01;
    int n = 2/del + 1;
    MPI_Init(&argc, &argv);
    jacobi(del, n);
    MPI_Finalize();
    return 0;
}
