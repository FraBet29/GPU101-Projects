#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <thrust/count.h>

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

    thrust::fill(colors, colors + n, -1); // init colors to -1

    for(int c = 0; c < n; c++) {
        int nt = 256;
        int nb = ((n + nt - 1)/nt);
        color_jpl_kernel<<<nb, nt>>>(n, c, 
                                     Ao, Ac, Av, 
                                     randoms, 
                                     colors);
        int left = (int)thrust::count(colors, colors + n, -1);
        if (left == 0) break;
    }
}

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

// GPU implementation of SYMGS using CSR
void symgs_csr_sw_parallel(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal, const int num_vals)
{   

    printf("Number of rows: %d\n", num_rows);
    printf("Number of values: %d\n", num_vals);

    int *d_col_ind;
    float *d_values, *d_x;

    check_call(cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int)));
    check_call(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
    check_call(cudaMalloc((void**)&d_x, num_rows * sizeof(float)));
    check_call(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_x, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // We will need to update x at each iteration

    // CUDA kernel parameters
    unsigned int threads_per_block = 64;
    unsigned int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block; // The GPU processes at most row_values values per iteration (worst case)

    printf("Number of blocks and threads per block: %d, %d\n", num_blocks, threads_per_block);

    // Allocate arrays to perform reduction (each element will contain the result computed by one GPU block)
    // The number of blocks effectively needed varies at each iteration, here we consider the worst case to perform one memory allocation once for all
    float *h_red = (float *)malloc(num_blocks * sizeof(float));
    float *d_red;

    check_call(cudaMalloc((void**)&d_red, num_blocks * sizeof(float)));

    // Forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        printf("Processing values from %dth to %dth\n", row_start, row_end);

        // Compute the number of blocks effectively needed
        unsigned int num_blocks_eff = ((row_end - row_start) + threads_per_block - 1) / threads_per_block;

        printf("Number of blocks effectively needed: %d\n", num_blocks_eff);

        reduction<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_col_ind, d_values, d_x, d_red, row_start, row_end);
        check_kernel_call();
        cudaDeviceSynchronize();

        // Copy the elements which have been effectively processed
        check_call(cudaMemcpy(h_red, d_red, num_blocks_eff * sizeof(float), cudaMemcpyDeviceToHost));

        // Complete reduction
        for(int j = 0; j < num_blocks_eff; ++j)
        {
            sum -= h_red[j];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;

        check_call(cudaMemcpy(d_x + i, x + i, sizeof(float), cudaMemcpyHostToDevice)); // Update current x
    }

    // Backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        // Compute the number of blocks effectively needed
        unsigned int num_blocks_eff = ((row_end - row_start) + threads_per_block - 1) / threads_per_block;

        reduction<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(d_col_ind, d_values, d_x, d_red, row_start, row_end);
        check_kernel_call();
        cudaDeviceSynchronize();

        // Copy the elements which have been effectively processed
        check_call(cudaMemcpy(h_red, d_red, num_blocks_eff * sizeof(float), cudaMemcpyDeviceToHost));

        // Complete reduction
        for(int j = 0; j < num_blocks_eff; ++j)
        {
            sum -= h_red[j];
        }
        
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;

        check_call(cudaMemcpy(d_x + i, x + i, sizeof(float), cudaMemcpyHostToDevice)); // Update current x
    }

    free(h_red);
    check_call(cudaFree(d_col_ind));
    check_call(cudaFree(d_values));
    check_call(cudaFree(d_x));
    check_call(cudaFree(d_red));

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

    int *colors = (int *)malloc(num_vals * sizeof(int));

    color_jpl(num_vals, row_ptr, col_ind, values, colors);

    for(int i = 0; i < num_vals; ++i)
    {
        printf("%d", colors[i]);
    }
    printf("\n");

    /*
    start_gpu = get_time();
    symgs_csr_sw_parallel(row_ptr, col_ind, values, num_rows, x_par, matrixDiagonal, num_vals);
    end_gpu = get_time();

    // Print GPU time
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

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

    return 0;
}

// CHECK IF THE MATRIX FITS INTO THE GPU
// SHARED MEMORY IMPLEMENTATION?
