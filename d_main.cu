#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h"
#include "molecule.h"
#include "d_main.h"

__global__ void d_discombulateKernel(float * energyGrid, dim3 grid, float gridspacing,
                                     const float *atoms, int numatoms);

/* 
    d_main.cu 
    Calculates an electrostatic potential grid for the molecule passed. 
    
    energyGrid: The grid that will contain the result
    atoms: An array of all atoms of the molecule and their positions. 
    numAtoms: The number of atoms in the molecule.
*/
void d_discombobulate(float * energyGrid, int dimX, int dimY, int dimZ, float gridSpacing, float *atoms, int numAtoms) {
    /* 
        TODO: Write host code to set up the cuda kernel, and launch d_discombobulateKernel
    */
    int gridSize = sizeof(float) * dimX * dimY * dimZ;
    float * d_energyGrid;
    CHECK(cudaMalloc((void**)&d_energyGrid, gridSize));

    float * d_atoms;
    CHECK(cudaMalloc((void**)&d_atoms, numAtoms * 4 * sizeof(int)));
    CHECK(cudaMemcpy(d_atoms, atoms, numAtoms * 4 * sizeof(int), cudaMemcpyHostToDevice));

    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(ceil((1.0 * dimZ) / 1024), 1, 1);

    dim3 grid(dimX, dimY, dimZ);

    d_discombulateKernel<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, d_atoms, numAtoms);

    CHECK(cudaMemcpy(energyGrid, d_energyGrid, gridSize, cudaMemcpyDeviceToHost));



}

/* 
    d_discombobulateKernel
    A kernel that calculates the electrostatic potential and stores it 
    in a float array

    energyGrid: The float array associated with the molecule 
*/
__global__ void d_discombulateKernel(float * energyGrid, dim3 grid, float gridSpacing,
                                     const float *atoms, int numAtoms) {
    /* 
        TODO: Write code to calculate the energy grid and store it in the 
        float array energyGrid. 
    */

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