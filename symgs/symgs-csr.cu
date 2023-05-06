#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

#define check_call(call)                                     \
{                                                            \
  const cudaError_t err = call;                              \
  if (err != cudaSuccess) {                                  \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), \
                                    __FILE__, __LINE__);     \
    exit(EXIT_FAILURE);                                      \
  }                                                          \
}

#define check_kernel_call()                                  \
{                                                            \
  const cudaError_t err = cudaGetLastError();                \
  if (err != cudaSuccess) {                                  \
    printf("%s in %s at line %d\n", cudaGetErrorString(err), \
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
    //int err;
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

// Graph coloring algorithm
// SOURCE: https://developer.nvidia.com/blog/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
__global__ void color_jpl_kernel(int n, int c, const int *Ao, 
                                 const int *Ac, const float *Av, 
                                 const int *randoms, int *colors)
{   
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; 
       i < n; 
       i += blockDim.x * gridDim.x) 
  {   
    bool f = true; // true iff you have max random

    // ignore nodes colored earlier
    if ((colors[i] != -1)) continue; 

    int ir = randoms[i];

    // look at neighbors to check their random number
    for (int k = Ao[i]; k < Ao[i+1]; k++) {        
      // ignore nodes colored earlier (and yourself)
      int j = Ac[k];
      int jc = colors[j];
      if (((jc != -1) && (jc != c)) || (i == j)) continue; 
      int jr = randoms[j];
      if (ir <= jr) f = false;
    }

    // assign color if you have the maximum random number
    if (f) colors[i] = c;
  }
}

void color_jpl(int n, 
               const int *Ao, const int *Ac, const float *Av, 
               int *colors) 
{   
    int *randoms = (int *)malloc(n * sizeof(int)); // allocate and init random array

    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        randoms[i] = (rand() % 100);
    }

    int *d_randoms;

    check_call(cudaMalloc((void **)&d_randoms, n * sizeof(int)));
    check_call(cudaMemcpy(d_randoms, randoms, n * sizeof(int), cudaMemcpyHostToDevice));

    for (int i = 0; i < n; i++)
    {
        colors[i] = -1; // init colors to -1
    }

    int *d_colors;

    check_call(cudaMalloc((void **)&d_colors, n * sizeof(int)));
    check_call(cudaMemcpy(d_colors, colors, n * sizeof(int), cudaMemcpyHostToDevice));

    printf("Colors initialization ok.\n");

    for(int c = 0; c < n; c++) {
        int nt = 256;
        int nb = ((n + nt - 1)/nt);
        color_jpl_kernel<<<nb, nt>>>(n, c, 
                                     Ao, Ac, Av, 
                                     d_randoms, 
                                     d_colors);
        check_kernel_call();
        cudaDeviceSynchronize();
        check_call(cudaMemcpy(colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost));
        int left = 0;
        for (int i = 0; i < n; i++)
        {
            if (colors[i] == -1)
            {
                left = 1;
                break;
            }
        }
        if (left == 0) break;
    }
}

/*
// Parallel reduction
__global__ void reduction(const int *col_ind, const float *values, const float *x, float *red, const int row_start, const int row_end)
{
    extern __shared__ float sum[]; // Array of size blockDim.x (i.e. threads_per_block)

    unsigned int i = threadIdx.x; // Local thread ID
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x; // Global thread ID (potentially ranging from 0 to num_rows)

    if(tid > row_start && tid < row_end)
    {
        sum[i] = values[tid] * x[col_ind[tid]]; // Compute all the values to be added
    }
    else
    {
        sum[i] = 0.0f;
    }
    __syncthreads();

    for(unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	  {	
		    if(i < s)
        {
			      sum[i] += sum[i + s];
		    }
		    __syncthreads();
    }

    if(i == 0)
    {
        red[blockIdx.x] = sum[0];
    }
}
*/

// GPU implementation of SYMGS using CSR
__global__ void symgs_csr_sw_parallel_fw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const int num_vals, float *x, const float *matrixDiagonal, int *x_flags)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        x_flags[i] = 0;
    }
    __syncthreads();

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = i; j < row_end; j++) // Old values
        {
            sum -= values[j] * x[col_ind[j]];
        }

        for (int j = row_start; j < i; j++) // New values
        {   
            if (x_flags[col_ind[j]] == 1)
                sum -= values[j] * x[col_ind[j]];
            else
                j--; // Wait until the needed value of x is updated
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
        x_flags[i] = 1;
    }
}

__global__ void symgs_csr_sw_parallel_bw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, const int num_vals, float *x, const float *matrixDiagonal, int *x_flags)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        x_flags[i] = 0;
    }
    __syncthreads();

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < i; j++) // Old values
        {
            sum -= values[j] * x[col_ind[j]];
        }
        
        for (int j = i; j < row_end; j++) // New values
        {   
            if (x_flags[col_ind[j]] == 1)
                sum -= values[j] * x[col_ind[j]];
            else
                j--; // Wait until the needed value of x is updated
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
        x_flags[i] = 1;
    }
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

    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // Print CPU time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);

    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_matrixDiagonal, *d_x;
    int *d_x_flags;

    check_call(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    check_call(cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int)));
    check_call(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
    check_call(cudaMalloc((void**)&d_matrixDiagonal, num_rows * sizeof(float)));
    check_call(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    check_call(cudaMalloc((void**)&d_x_flags, num_rows * sizeof(int)));

    check_call(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_matrixDiagonal, d_matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // Use the same random vector for the parallel version

    printf("CUDA setup ok.\n");

    int *colors = (int *)malloc(num_vals * sizeof(int));

    color_jpl(num_vals, d_row_ptr, d_col_ind, d_values, colors);

    for(int i = 0; i < num_vals; ++i)
    {
        printf("%d", colors[i]);
    }
    printf("\n");
    
    /*
    // CUDA kernel parameters
    unsigned int threads_per_block = 1024;
    unsigned int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    start_gpu = get_time();

    symgs_csr_sw_parallel_fw<<<num_blocks, threads_per_block>>>(d_row_ptr, d_col_ind, d_values, num_rows, num_vals, d_x, d_matrixDiagonal, d_x_flags); // Forward sweep
    check_kernel_call();
    cudaDeviceSynchronize();

    symgs_csr_sw_parallel_bw<<<num_blocks, threads_per_block>>>(d_row_ptr, d_col_ind, d_values, num_rows, num_vals, d_x, d_matrixDiagonal, d_x_flags); // Backward sweep
    check_kernel_call();
    cudaDeviceSynchronize();

    end_gpu = get_time();

    // Print GPU time
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

    check_call(cudaMemcpy(x_par, d_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

    printf("x: %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4]);
    printf("x par: %f %f %f %f %f\n", x_par[0], x_par[1], x_par[2], x_par[3], x_par[4]);

    // Compare results
    float err = inf_err(x, x_par, num_rows);
    printf("Error: %f\n", err);
    if(err > 1e-4)
    {
        printf("CPU and GPU give different results (error > threshold)\n");
    }
    else
    {
        printf("CPU and GPU give similar results (error < threshold)\n");
    }
    */

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(x_par);
    free(colors);
    check_call(cudaFree(d_row_ptr));
    check_call(cudaFree(d_col_ind));
    check_call(cudaFree(d_values));
    check_call(cudaFree(d_matrixDiagonal));
    check_call(cudaFree(d_x));

    return 0;
}

// CHECK IF THE MATRIX FITS INTO THE GPU
// SHARED MEMORY IMPLEMENTATION?
