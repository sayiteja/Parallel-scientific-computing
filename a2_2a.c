#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define MAX_ITERATIONS 100000
#define TOLERANCE 1e-4
#define PI 3.14159265359

double q(int a, int b, double del){
    double x = a*del - 1;
    double y = b*del - 1;
    return x*x - y*y;
}

double norm(double **A, double **B, int n){
    double sum = 0.0;
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            sum += (A[i][j] - B[i][j])*(A[i][j] - B[i][j]);
        }
    }
    return sqrt(sum);
}

double **jacobi(double** A, double del, int n){
    A = (double **)malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++){
        A[i] = (double *)malloc(n * sizeof(double));
    }
    double** A_old = (double **)malloc(n * sizeof(double *));
    for(int i = 0; i < n; i++){
        A_old[i] = (double *)malloc(n * sizeof(double));
    }

    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
                A[i][j] = 0.0;
                if(j==0) A[i][0] = sin(2*PI*(i*del-1));
                if(j==n-1) A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
        }
    }

    int iter = 0;
    double error;
    error = 1.0;
    while (iter < MAX_ITERATIONS && error > TOLERANCE){
        for(int i = 0; i < n; i++){
            for (int j = 0; j < n; j++){
                A_old[i][j] = A[i][j];
            }
        }
        for(int i = 1; i < n-1; i++){
            for (int j = 1; j < n-1; j++){
                A[i][j] = 0.25*(A_old[i+1][j]+A_old[i-1][j]+A_old[i][j+1]+A_old[i][j-1]) + 0.25*del*del*q(i,j,del);
            }
        }
        for(int i = 0; i < n; i++){
            A[i][n-1] = (4*A[i][n-2] - A[i][n-3])/3;
        }
        error = norm(A, A_old, n);
        printf("iter: %d, err: %f\n", iter, error);
        iter++;
    }
    free(A_old);
    return A;
}

int main(){
    double **A;
    double del = 0.1;
    int n = 2/del + 1;
    clock_t start = clock();
    A = jacobi(A, del, n);
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Time spent: %f\n", time_spent);
    free(A);
}