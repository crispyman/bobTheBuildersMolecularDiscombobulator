#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h"
#include "molecule.h"
#include "d_main.h"

__global__ void d_discombulateKernel(float * energyGrid, const float *atoms, dim3 grid, float gridspacing,
                                      int numatoms);

__global__ void d_discombulateKernelConst(float * energyGrid, dim3 grid, float gridSpacing,
                                          int numAtoms);

/* 
    d_main.cu 
    Calculates an electrostatic potential grid for the molecule passed. 
    
    energyGrid: The grid that will contain the result
    atoms: An array of all atoms of the molecule and their positions. 
    numAtoms: The number of atoms in the molecule.
*/
__constant__ float constAtoms[1024];

int d_discombobulate(float * energyGrid, float *atoms, int dimX, int dimY, int dimZ, float gridSpacing,  int numAtoms, int which){
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));
    CHECK(cudaEventRecord(start_gpu));

    /*
     * TODO: Find out why cudaMemcpyToSymbol is silently failing
     */

    CHECK(cudaMemcpyToSymbol(constAtoms, atoms, sizeof(float) * numAtoms * 4));



    int gridSize = sizeof(float) * dimX * dimY * dimZ;
    float * d_energyGrid;
    CHECK(cudaMalloc((void**)&d_energyGrid, gridSize));

    float * d_atoms;
    CHECK(cudaMalloc((void**)&d_atoms, numAtoms * 4 * sizeof(int)));
    CHECK(cudaMemcpy(d_atoms, atoms, numAtoms * 4 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 grid(dimX, dimY, dimZ);

    if (which == 0) {
        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimZ) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernel<<<gridDim, blockDim>>>(d_energyGrid, d_atoms, grid, gridSpacing, numAtoms);
    }
    if (which == 1) {
        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimZ) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernelConst<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtoms);
    }

    CHECK(cudaMemcpy(energyGrid, d_energyGrid, gridSize, cudaMemcpyDeviceToHost));

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;

}

/* 
    d_discombobulateKernel
    A kernel that calculates the electrostatic potential and stores it 
    in a float array

    energyGrid: The float array associated with the molecule 
*/
__global__ void d_discombulateKernel(float * energyGrid, const float *atoms, dim3 grid, float gridSpacing,
                                     int numAtoms) {


    int i, j, n;
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.z) {
        float z = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);
        int atomArrDim = numAtoms * 4;
        for (j = 0; j < grid.y; j++) {
            float y = gridSpacing * (float) j;
            for (i = 0; i < grid.x; i++) {
                float x = gridSpacing * (float) i;
                float energy = 0.0f;
                for (n = 0; n < atomArrDim; n += 4) {
                    float dx = x - atoms[n];
                    float dy = y - atoms[n + 1];
                    float dz = z - atoms[n + 2];
                    energy += atoms[n + 3] / sqrt(dx * dx + dy * dy + dz * dz);
                }
                energyGrid[grid.x * grid.y * (blockIdx.x * blockDim.x + threadIdx.x) + grid.x * j + i] = energy;
            }
        }
    }
}

__global__ void d_discombulateKernelConst(float * energyGrid, dim3 grid, float gridSpacing,
                                     int numAtoms) {


    int i, j, n;
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.z) {
        float z = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);
        int atomArrDim = numAtoms * 4;
        for (j = 0; j < grid.y; j++) {
            float y = gridSpacing * (float) j;
            for (i = 0; i < grid.x; i++) {
                float x = gridSpacing * (float) i;
                float energy = 0.0f;
                for (n = 0; n < atomArrDim; n += 4) {
                    float dx = x - constAtoms[n];
                    float dy = y - constAtoms[n + 1];
                    float dz = z - constAtoms[n + 2];
                    energy += constAtoms[n + 3] / sqrt(dx * dx + dy * dy + dz * dz);
                }
                energyGrid[grid.x * grid.y * (blockIdx.x * blockDim.x + threadIdx.x) + grid.x * j + i] = energy;
            }
        }
    }
}