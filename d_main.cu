#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "config.h"
#include "molecule.h"
#include "d_main.h"

__global__ void d_discombulateKernel(float * energyGrid, const atom *atoms, dim3 grid, float gridspacing,
                                      int numatoms);

__global__ void d_discombulateKernelConst(float * energyGrid, dim3 grid, float gridSpacing,
                                          int numAtoms);

static __global__ void emptyKernel();

/* 
    d_main.cu 
    Calculates an electrostatic potential grid for the molecule passed. 
    
    energyGrid: The grid that will contain the result
    atoms: An array of all atoms of the molecule and their positions. 
    numAtoms: The number of atoms in the molecule.
*/
__constant__ atom constAtoms[256];
int d_discombobulate(float * energyGrid, atom * atoms, int dimX, int dimY, int dimZ, float gridSpacing,  int numAtoms, int which){
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;;




    CHECK(cudaEventCreate(&start_gpu));
    CHECK(cudaEventCreate(&stop_gpu));

    emptyKernel<<<1024, 1024>>>();


    CHECK(cudaEventRecord(start_gpu));




    int gridSize = sizeof(float) * dimX * dimY * dimZ;
    float * d_energyGrid;
    CHECK(cudaMalloc((void**)&d_energyGrid, gridSize));



    dim3 grid(dimX, dimY, dimZ);

    if (which == 0) {
        atom * d_atoms;
        CHECK(cudaMalloc((void**)&d_atoms, numAtoms * sizeof(atom)));
        CHECK(cudaMemcpy(d_atoms, atoms, numAtoms * sizeof(atom), cudaMemcpyHostToDevice));

        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernel<<<gridDim, blockDim>>>(d_energyGrid, d_atoms, grid, gridSpacing, numAtoms);


        CHECK(cudaFree(d_atoms));
    }
    if (which == 1) {
        CHECK(cudaMemcpyToSymbol(constAtoms, atoms, sizeof(float) * numAtoms * 4));

        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernelConst<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtoms);
    }

    CHECK(cudaMemcpy(energyGrid, d_energyGrid, gridSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(d_energyGrid));

    CHECK(cudaEventRecord(stop_gpu));
    CHECK(cudaEventSynchronize(stop_gpu));
    CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));
    return gpuMsecTime;

}

/* 
    d_discombobulateKernel
    A kernel that calculates the electrostatic potential and stores it 
    in a float array

    This performed about 7x better than cpu before switching to atom struct

    energyGrid: A float array of the sample points in and around the molecule
    atoms: an array of structs that have all the necessary information about each atom, see molecule.h
    grid: a dim3 struct containing the dimensions of energyGrid
    gridSpacing: the space between grid points along each axis
    numAtoms: number of atoms in atoms array
*/
__global__ void d_discombulateKernel(float * energyGrid, const atom *atoms, dim3 grid, float gridSpacing,
                                     int numAtoms) {


    int i, j, n;
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.x) {
        float x = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);

            for (i = 0; i < grid.z; i++) {
                float z = gridSpacing * (float) i;
                for (j = 0; j < grid.y; j++) {
                    float y = gridSpacing * (float) j;
                float energy = 0.0f;
                for (n = 0; n < numAtoms; n ++) {
                    float dx = x - atoms[n].x;
                    float dy = y - atoms[n].y;
                    float dz = z - atoms[n].z;
                    energy += atoms[n].charge / sqrtf(dx * dx + dy * dy + dz * dz);
                }
                    __syncthreads();
                    energyGrid[grid.x * grid.y * i + grid.x * j + (blockIdx.x * blockDim.x + threadIdx.x)] = energy;

            }
        }
    }
}

/*
 * d_discombobulateKernel
 * A kernel that calculates the electrostatic potential and stores it in a float array,
    this version uses constant memory instead of passing in an atoms array
 *
 * energyGrid: A float array of the sample points in and around the molecule
 * grid: a dim3 struct containing the dimensions of energyGrid
 * gridSpacing: the space between grid points along each axis
 *   numAtoms: number of atoms in constAtoms
 */

__global__ void d_discombulateKernelConst(float * energyGrid, dim3 grid, float gridSpacing,
                                     int numAtoms) {


    int i, j, n;
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.x) {
        float x = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);

            for (i = 0; i < grid.z; i++) {
                float z = gridSpacing * (float) i;
                for (j = 0; j < grid.y; j++) {
                    float y = gridSpacing * (float) j;
                float energy = 0.0f;
                for (n = 0; n < numAtoms; n ++) {
                    float dx = x - constAtoms[n].x;
                    float dy = y - constAtoms[n].y;
                    float dz = z - constAtoms[n].z;
                    float charge = constAtoms[n].charge;
                    energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
                }
                    __syncthreads();
                    energyGrid[grid.x * grid.y * i + grid.x * j + (blockIdx.x * blockDim.x + threadIdx.x)] = energy;

            }
        }
    }
}

// an empty kernel to improve timing?
__global__ void emptyKernel()
{
}