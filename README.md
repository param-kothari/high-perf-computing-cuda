## Overview

In this project, we measure the time taken to compute the prices of 5 different European call options using Monte Carlo simulations. For the purpose of this project, we write two separate solutions: one using CUDA (GPU + CPU) and one using OpenMP multithreading in C++ (CPU only). I have included a Makefile which runs the CUDA solution, but I have added the C++ solution in the zipped folder as well for reference. I used the following methods in the CUDA solution to make the code faster:

### Parallel computing using GPU

In the code, we first create multiple arrays on the host side storing the option parameters. We then use these arrays to allocate memory on the GPU and subsequently copy the parameters data to the GPU. Using 128 threads per block, we create the required blocks on the GPU side using the "dim" command. Then, we use these threads to parallely compute the results for each option. Each thread computes one Monte Carlo path simulation per option parallely. We create a separate prices array to store the results on the device side and then later copy the results to an array defined on the host side. The pricing function is done on the device specified using the "**global**" command.

### Random number generation using cuRAND

Generating 5 million copies of random normally distributed numbers on the host side using C++ libraries was very costly. Doing that took us close to a full second. The second approach I thought of was to use the Intel MKL to generate a stream of random normally distributed numbers on the host side and then copy them to the GPU side. This approach was faster but still took ~0.1 seconds to do so since we would have to copy 5 million data points to GPU which is an expensive operation. Thus, we use cuRAND defined in CUDA to generate the random numbers. We use the following code to populate the "d_z" array in the device with random normally distributed numbers with mean 0 and std dev 1:

```cpp
curandGenerator_t gen;
curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
curandSetPseudoRandomGeneratorSeed(gen, 42);
curandGenerateNormal(gen, d_z, M, 0.0f, 1.0f);
```

### Parallel reduction and shared memory

In our original approach, each thread was used to compute the price of a single path for a single option and each of these values were summed up. Each thread was using the global memory which was making it slower. Instead, for the purpose of parallel reduction, we define a shared data array for quicker access to each thread. This shared data array holds results of each thread per block. Then, using this shared data, we perform sequential addressing reduction to reduce the number of participating threads in the accumulation process. At each step, we add the results of the second half of the array to the first half. That way, in the end, the total accumulated result is stored in the 0th index.

Reference: [NVIDIA Mark Harris Article Link](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf)

## Files

- `main.cu`: The main CUDA source code file.
- `main.cpp`: C++ source code file; alternate soltuion.
- `Makefile`: The Makefile to compile and run the program.

## Usage

1. I worked with a GPU node with 2 CPU cores to drive the GPU cores. For this project, I was granted Midway GPU access from the University of Chicago.

2. Make sure that the required CUDA module and dependencies are installed.

3. Run the Makefile. Upon doing so, the output of the program will be displayed in the terminal.

```bash
make
```

### Alternate Solution Usage

1. For this, I used a computing node with 12 cores.

2. Make sure that the required Intel and MKL modules are installed.

3. Compile the program.

```bash
icc -std=c++17 main.cpp -o main -qmkl -qopenmp
```

4. Run the program. Upon doing so, the output of the program will be displayed in the terminal.

```bash
./main
```
