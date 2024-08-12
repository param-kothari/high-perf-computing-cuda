// Headers
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda.h>
#include <curand.h>

// Namespaces
using namespace std;
using namespace std::chrono;

// Function to compute call option prices using monte carlo simulations on the GPU side
__global__ void monte_carlo_option_pricing(float *S0, float *K, float *T, float *v, float *r, float *C, float *z, int M)
{
    // Define a shared array of floats for quicker memory access
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int opt = blockIdx.y;

    // Compute the option payoff using the thread ID and block ID
    float payoff = 0.0f;
    if (idx < M)
    {
        float ST = S0[opt] * __expf((r[opt] - v[opt] * v[opt] * 0.5) * T[opt] + v[opt] * z[idx] * __fsqrt_rn(T[opt]));
        payoff = __expf(-r[opt] * T[opt]) * max(ST - K[opt], 0.0f) / M;
    }

    sdata[tid] = payoff;
    __syncthreads(); // Synchronize threads before moving on to the next part

    // Parallel reduction using sequential addressing to optimize the process
    // Ref: https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // The accumulated results are stored in 0th index. We use atomicAdd to safely add this value to the correct C index
    if (tid == 0)
    {
        atomicAdd(&C[opt], sdata[0]);
    }
}

int main()
{
    // Reset the device and free up any space
    cudaDeviceReset();
    cudaFree(0);

    auto t1 = high_resolution_clock::now();

    static const int M = 5000000;
    static const int numOptions = 5;

    // Define the number of threads to be used and compute the blocks needed using a ceiling division
    const int threads = 128;
    const int blocks = (M + threads - 1) / threads; // 39063 blocks for our purpose

    float *S0 = new float[numOptions];
    float *K = new float[numOptions];
    float *T = new float[numOptions];
    float *v = new float[numOptions];
    float *r = new float[numOptions];
    float *C = new float[numOptions];

    // Allocate required GPU memory
    float *d_S0, *d_K, *d_T, *d_v, *d_r, *d_C, *d_z;
    cudaMalloc((float **)&d_S0, numOptions * sizeof(float));
    cudaMalloc((float **)&d_K, numOptions * sizeof(float));
    cudaMalloc((float **)&d_T, numOptions * sizeof(float));
    cudaMalloc((float **)&d_v, numOptions * sizeof(float));
    cudaMalloc((float **)&d_r, numOptions * sizeof(float));
    cudaMalloc((float **)&d_C, numOptions * sizeof(float));
    cudaMalloc((float **)&d_z, M * sizeof(float));

    /*
    Initialize the options given as (with K = 100 for all cases):
        a)   S = 90;  r = 0.01; v = 0.3, T = 1 (year), N = 5 million
        b)   S = 95; r = 0.02,  v= 0.3;  T = 1.2 (years),  N = 5 million
        c)   S = 100;  r = 0.03; v = 0.3;  T = 1.5 (years), N = 5 million
        d)   S = 105; r = 0.04,  v= 0.3;  T = 2 (years), N = 5 million
        e)   S = 110; r = 0.05,  v= 0.3;  T = 2.5 (years), N = 5 million
    */

    for (int i = 0; i < 5; i++)
    {
        S0[i] = 90.0f + 5.0f * i;
        r[i] = 0.01f + 0.01f * i;
        v[i] = 0.3f;
        K[i] = 100.0f;
    }

    T[0] = 1.0f;
    T[1] = 1.2f;
    T[2] = 1.5f;
    T[3] = 2.0f;
    T[4] = 2.5f;

    for (int i = 0; i < numOptions; i++)
    {
        C[i] = 0.0f;
    }

    // Copy the host data to device
    cudaMemcpy(d_S0, S0, numOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, numOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_T, T, numOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, numOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r, numOptions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, numOptions * sizeof(float), cudaMemcpyHostToDevice);

    // Use the curand random number generator to generate M number of normally distributed floats
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 42);
    curandGenerateNormal(gen, d_z, M, 0.0f, 1.0f);

    // Define the grid of threads
    dim3 grid(blocks, numOptions);
    monte_carlo_option_pricing<<<grid, threads, threads * sizeof(float)>>>(d_S0, d_K, d_T, d_v, d_r, d_C, d_z, M);
    cudaDeviceSynchronize();

    // Copy the device results data back to the host
    cudaMemcpy(C, d_C, numOptions * sizeof(float), cudaMemcpyDeviceToHost);

    auto t2 = high_resolution_clock::now();

    for (int i = 0; i < numOptions; i++)
    {
        cout << "Option Price " << char('a' + i) << "): " << C[i] << endl;
    }

    cout << "\nTime taken to price the given " << numOptions << " options is: " << duration_cast<milliseconds>(t2 - t1).count() << " ms" << endl;

    // Free the GPU/CPU memory used
    cudaFree(d_S0);
    cudaFree(d_K);
    cudaFree(d_T);
    cudaFree(d_v);
    cudaFree(d_r);
    cudaFree(d_C);
    cudaFree(d_z);

    curandDestroyGenerator(gen);

    delete[] S0;
    delete[] K;
    delete[] T;
    delete[] v;
    delete[] r;
    delete[] C;

    return 0;
}
