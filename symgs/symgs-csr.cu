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

/*
// Graph coloring algorithm
// SOURCE: https://developer.nvidia.com/blog/graph-coloring-more-parallelism-for-incomplete-lu-factorization/
__global__ void color_jpl_kernel(int n, int c, const int *Ao, const int *Ac, const float *Av, const int *randoms, int *colors)
{   
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) 
    {   
        bool f = true; // true iff you have max random

        if ((colors[i] != -1)) continue; // ignore nodes colored earlier

        int ir = randoms[i];

        // look at neighbors to check their random number
        for (int k = Ao[i]; k < Ao[i+1]; k++)
        {
          // ignore nodes colored earlier (and yourself)
          int j = Ac[k];
          int jc = colors[j];
          if (((jc != -1) && (jc != c)) || (i == j)) continue; 
          int jr = randoms[j];
          if (ir <= jr) f = false;
        }

        if (f) colors[i] = c; // assign color if you have the maximum random number
    }
}

// Ao contains the matrix row offsets, Ac contains the column indices, and Av contains the non-zero values
void color_jpl(int n, const int *Ao, const int *Ac, const float *Av, int *colors) 
{   
    int *randoms = (int *)malloc(n * sizeof(int)); // allocate and init random array

    srand(time(NULL));
    for (int i = 0; i < n; i++)
    {
        randoms[i] = (rand() % 1000);
    }

    for (int i = 0; i < n; i++)
        {
            printf("%d ", randoms[i]);
        }
        printf("\n");

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

    for(int c = 0; c < n; c++)
    {
        int nt = 256;
        int nb = (n + nt - 1) / nt;
        color_jpl_kernel<<<nb, nt>>>(n, c, Ao, Ac, Av, d_randoms, d_colors);
        check_kernel_call();
        cudaDeviceSynchronize();
        check_call(cudaMemcpy(colors, d_colors, n * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; i++)
        {
            printf("%d ", colors[i]);
        }
        printf("\n");
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
*/

__global__ void my_color(int num_rows, const int *row_ptr, const int *col_ind, int *colors_fw, int *colors_bw)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];

        int curr = -1;
        int prev = -1;

        for (int j = row_start; j < row_end; j++)
        {
            prev = curr;
            curr = col_ind[j];
            if (curr >= i) break;
        }

        colors_fw[i] = prev + 1;

        curr = num_rows;
        prev = num_rows;

        for (int j = row_end - 1; j > -1; j--)
        {
            prev = curr;
            curr = col_ind[j];
            if (curr <= i) break;
        }

        colors_bw[i] = prev - 1;
    }
}

__global__ void symgs_csr_sw_colors(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x_old, float *x_new, float *matrixDiagonal, const int *colors, const int color)
{   
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    for (int i = idx; i < num_rows; i += gridDim.x * blockDim.x)
    {
        if (colors[i] == color)
        {   
            float sum = x_new[i];
            const int row_start = row_ptr[i];
            const int row_end = row_ptr[i + 1];
            float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

            for (int j = row_start; j < row_end; j++)
            {
                sum -= values[j] * x_new[col_ind[j]];
            }

            sum += x_new[i] * currentDiagonal; // Remove diagonal contribution from previous loop

            x_old[i] = sum / currentDiagonal;
        }
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
    float *d_x_old, *d_x_new;

    check_call(cudaMalloc((void**)&d_x_old, num_rows * sizeof(float)));
    check_call(cudaMalloc((void**)&d_x_new, num_rows * sizeof(float)));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that'offset the reason of the +1
    }

    check_call(cudaMemcpy(d_x_old, x, num_rows * sizeof(float), cudaMemcpyHostToDevice)); // Use the same random vector for the parallel version
    check_call(cudaMemcpy(d_x_new, x, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    memcpy(x_par, x, num_rows * sizeof(float));

    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // Print CPU time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);

    int *d_row_ptr, *d_col_ind;
    float *d_values, *d_matrixDiagonal;

    check_call(cudaMalloc((void**)&d_row_ptr, (num_rows + 1) * sizeof(int)));
    check_call(cudaMalloc((void**)&d_col_ind, num_vals * sizeof(int)));
    check_call(cudaMalloc((void**)&d_values, num_vals * sizeof(float)));
    check_call(cudaMalloc((void**)&d_matrixDiagonal, num_rows * sizeof(float)));

    check_call(cudaMemcpy(d_row_ptr, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_col_ind, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_values, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    check_call(cudaMemcpy(d_matrixDiagonal, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));

    printf("CUDA setup ok.\n");

    // CUDA kernel parameters
    unsigned int threads_per_block = 1024;
    unsigned int num_blocks = (num_rows + threads_per_block - 1) / threads_per_block;

    start_gpu = get_time();
    
    int *colors_fw = (int *)malloc(num_rows * sizeof(int));
    int *colors_bw = (int *)malloc(num_rows * sizeof(int));
    
    int *d_colors_fw, *d_colors_bw;

    check_call(cudaMalloc((void **)&d_colors_fw, num_rows * sizeof(int)));
    check_call(cudaMalloc((void **)&d_colors_bw, num_rows * sizeof(int)));

    my_color<<<num_blocks, threads_per_block>>>(num_rows, d_row_ptr, d_col_ind, d_colors_fw, d_colors_bw);

    check_call(cudaMemcpy(colors_fw, d_colors_fw, num_rows * sizeof(int), cudaMemcpyDeviceToHost));
    check_call(cudaMemcpy(colors_bw, d_colors_bw, num_rows * sizeof(int), cudaMemcpyDeviceToHost));
    /*
    for (int i = 0; i < num_rows; ++i)
    {
        printf("%d ", colors[i]);
        if (i % 20 == 0) printf("\n");
    }
    printf("\n");
    */

    int num_colors_fw = 0;
    int num_colors_bw = 0;

    for (int i = 0; i < num_rows; ++i)
    {
        if (colors_fw[i] >= i) num_colors_fw++;
    }
    for (int i = num_rows - 1; i > -1; i--)
    {
        if (colors_bw[i] <= i) num_colors_bw++;
    }
    printf("Old vs new iterations fw: %d %d\n", num_rows, num_colors_fw);
    printf("Old vs new iterations bw: %d %d\n", num_rows, num_colors_bw);

    int num_blocks_fw = (num_rows - num_colors_fw + threads_per_block - 1) / threads_per_block;
    int num_blocks_bw = (num_rows - num_colors_bw + threads_per_block - 1) / threads_per_block;
    
    for (int i = 0; i < num_rows; i++)
    {
        symgs_csr_sw_colors<<<num_blocks, threads_per_block>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x_old, d_x_new, d_matrixDiagonal, d_colors_fw, i);
        check_kernel_call();
        cudaDeviceSynchronize();
        check_call(cudaMemcpy(&d_x_new[i], &d_x_old[i], sizeof(float), cudaMemcpyDeviceToDevice));
        /*
        check_call(cudaMemcpy(x_par, d_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k < num_rows; k++)
        {
            printf("%f ", x_par[k]);
        }
        printf("\n");
        */
    }
    for (int i = num_rows - 1; i >= 0; i--)
    {
        symgs_csr_sw_colors<<<num_blocks, threads_per_block>>>(d_row_ptr, d_col_ind, d_values, num_rows, d_x_old, d_x_new, d_matrixDiagonal, d_colors_bw, i);
        check_kernel_call();
        cudaDeviceSynchronize();
        check_call(cudaMemcpy(&d_x_new[i], &d_x_old[i], sizeof(float), cudaMemcpyDeviceToDevice));
        /*
        check_call(cudaMemcpy(x_par, d_x, num_rows * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k < num_rows; k++)
        {
            printf("%f ", x_par[k]);
        }
        printf("\n");
        */
    }

    end_gpu = get_time();

    // Print GPU time
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);

    check_call(cudaMemcpy(x_par, d_x_new, num_rows * sizeof(float), cudaMemcpyDeviceToHost));

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

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(x_par);
    free(colors_fw);
    free(colors_bw);
    check_call(cudaFree(d_row_ptr));
    check_call(cudaFree(d_col_ind));
    check_call(cudaFree(d_values));
    check_call(cudaFree(d_matrixDiagonal));
    check_call(cudaFree(d_x_old));
    check_call(cudaFree(d_x_new));
    check_call(cudaFree(d_colors_fw));
    check_call(cudaFree(d_colors_bw));

    return 0;
}

// IDEA: FIND INDEPENDENT SUBSETS OF VARIABLES VIA A GRAPH COLORING ALGORITHM, THEN APPLY GAUSS-SEIDEL ON EVERY SUBSET
// TWO VARIABLES HAVE THE SAME COLOR IF THEY ARE INDEPENDENT

// CHECK IF THE MATRIX FITS INTO THE GPU
// SHARED MEMORY IMPLEMENTATION?
