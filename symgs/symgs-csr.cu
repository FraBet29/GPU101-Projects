#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define check_call(call)                                     \
{                                                            \
  const cudaError_t err = call;                              \
  if (err != cudaSuccess) {                                  \
    printf("%offset in %offset at line %d\n", cudaGetErrorString(err), \
                                    __FILE__, __LINE__);     \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
}

#define check_kernel_call()                                  \
{                                                            \
  const cudaError_t err = cudaGetLastError();                \
  if (err != cudaSuccess) {                                  \
    printf("%offset in %offset at line %d\n", cudaGetErrorString(err), \
                                    __FILE__, __LINE__);     \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
}               

double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    int err;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int *row_ptr_t = (int *)malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *)malloc(*num_vals * sizeof(int));
    float *values_t = (float *)malloc(*num_vals * sizeof(float));
    float *matrixDiagonal_t = (float *)malloc(*num_rows * sizeof(float));
    // Collect occurrences of each row for determining the indices of row_ptr
    int *row_occurrences = (int *)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurrences[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurrences[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurrences[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurrences);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");
    
    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }

    fclose(file);
    
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}

// Parallel reduction
__global__ void reduction(const int *col_ind, const float *values, const float *x, float *sum, const int row_start, const int row_end)
{
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < row_end - row_start) // use only the memory that is needed
    {
        sum[tid] = values[row_start + tid] * x[col_ind[row_start + tid]];
    }

    for(unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{	
		if (tid < offset)
        {
			sum[tid] += sum[tid + offset];
		}
		__syncthreads();
    }
}

// GPU implementation of SYMGS using CSR
void symgs_csr_sw_parallel(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal, const int num_vals)
{   

    int *d_col_ind;
    float *d_values;

    // Allocate memory on the GPU once for all (even though we may be wasting some space)
    check_call(cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int)));
    check_call(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
    check_call(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));

    float *d_x;
    check_call(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    check_call(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // We will need to update x at each iteration

    float *d_sum;
    float sum_temp;

    // Allocate memory for the array used to perform the reduction on the GPU
    check_call(cudaMalloc((void**)&d_sum, num_vals * sizeof(float)));

    // CUDA kernel parameters
    unsigned int N = 256; // N <= 1024
    dim3 threads_per_block(N, 1, 1);
    dim3 num_blocks((num_rows + N - 1) / N, 1, 1); // The GPU processes at most num_rows values per iteration

    // Forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        reduction<<<num_blocks, threads_per_block>>>(d_col_ind, d_values, d_x, d_sum, row_start, row_end);
        check_kernel_call();
        cudaDeviceSynchronize();

        check_call(cudaMemcpy(&sum_temp, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        sum -= sum_temp;

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;

        check_call(cudaMemcpy(d_x + i, x + i, sizeof(float), cudaMemcpyHostToDevice)); // Update the current value
    }

    // Backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        check_call(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));

        reduction<<<num_blocks, threads_per_block>>>(d_col_ind, d_values, d_x, d_sum, row_start, row_end);
        check_kernel_call();
        cudaDeviceSynchronize();

        check_call(cudaMemcpy(&sum_temp, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
        sum -= sum_temp;

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;

        check_call(cudaMemcpy(d_x + i, x + i, sizeof(float), cudaMemcpyHostToDevice)); // Update the current value
    }

    check_call(cudaFree(d_col_ind));
    check_call(cudaFree(d_values));
    check_call(cudaFree(d_x));

}

float abs(const float x, const float y)
{
    if(x > y)
        return (x - y);
    else
        return (y - x);
}

float inf_err(const float *a, const float *b, int n)
{
    float err = 0.0f;

    for(int i = 0; i < n; ++i)
        err += abs(a[i] - b[i]);
    
    return err;
}

int main(int argc, const char *argv[])
{

    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

    double start_cpu, end_cpu;
    double start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    
    printf("Input file read.\n");
    
    float *x = (float *)malloc(num_rows * sizeof(float));
    float *x_par = (float *)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that'offset the reason of the +1
    }

    // Use the same random vector for the parallel version
    memcpy(x_par, x, num_rows * sizeof(float));

    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // Print CPU time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);

    start_gpu = get_time();
    symgs_csr_sw_parallel(row_ptr, col_ind, values, num_rows, x_par, matrixDiagonal, num_vals);
    end_gpu = get_time();

    // Print GPU time
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

    // Compare results
    float err = inf_err(x, x_par, num_rows * sizeof(float));
    if(err > 1e-4)
    {
        printf("CPU and GPU give different results (error > threshold)\n");
    }
    else
    {
        printf("CPU and GPU give similar results (error < threshold)\n");
    }

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(x_par);

    return 0;
}

// CHECK IF THE MATRIX FITS INTO THE GPU
// SHARED MEMORY IMPLEMENTATION?
