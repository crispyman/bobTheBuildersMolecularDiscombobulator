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

__global__ void d_discombulateKernelConst2D(float * energyGrid, dim3 grid, float gridSpacing,
                                            int numAtoms);

__global__ void d_discombulateKernelConst3D(float * energyGrid, dim3 grid, float gridSpacing,
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
    //zeros GPU memory since we want a zeroed energy grid to start with
    CHECK(cudaMemset(d_energyGrid, 0, gridSize));



    dim3 grid(dimX, dimY, dimZ);

    // 1D blocks no shared memory
    if (which == 0) {
        // copies atoms array to device memory
        atom * d_atoms;
        CHECK(cudaMalloc((void**)&d_atoms, numAtoms * sizeof(atom)));
        CHECK(cudaMemcpy(d_atoms, atoms, numAtoms * sizeof(atom), cudaMemcpyHostToDevice));

        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernel<<<gridDim, blockDim>>>(d_energyGrid, d_atoms, grid, gridSpacing, numAtoms);


        CHECK(cudaFree(d_atoms));
    }
        // 1D blocks shared memory
    else if (which == 1) {
        // copies atoms to shared memmory
        CHECK(cudaMemcpyToSymbol(constAtoms, atoms, sizeof(atom) * numAtoms));

        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernelConst<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtoms);
    }
    // 2D blocks shared memory
    else if (which == 2) {
        // copies atoms to shared memmory
        CHECK(cudaMemcpyToSymbol(constAtoms, atoms, sizeof(atom) * numAtoms));

        dim3 blockDim(THREADSPERBLOCK2D, THREADSPERBLOCK2D, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK2D), ceil((1.0 * dimY) / THREADSPERBLOCK2D), 1);
        d_discombulateKernelConst2D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtoms);
    }
    // 3D blocks shared memory
    else if (which == 3) {
        // copies atoms to shared memmory
        CHECK(cudaMemcpyToSymbol(constAtoms, atoms, sizeof(atom) * numAtoms));

        dim3 blockDim(THREADSPERBLOCK3D, THREADSPERBLOCK3D, THREADSPERBLOCK3D);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK3D), ceil((1.0 * dimY) / THREADSPERBLOCK3D), ceil((1.0 * dimZ) / THREADSPERBLOCK3D));
        d_discombulateKernelConst3D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtoms);
    }
    // Copies results to host
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
    // check to ensure thread is supposed to be doing work
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.x) {
        float x = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);

            for (i = 0; i < grid.z; i++) {
                float z = gridSpacing * (float) i;
                for (j = 0; j < grid.y; j++) {
                    float y = gridSpacing * (float) j;
                    // Generate gridGrid index
                    int gridIndex = grid.x * grid.y * i + grid.x * j + (blockIdx.x * blockDim.x + threadIdx.x);
                    // Load energyGrid value early
                    float energy = 0.0f;
                    float oldEnergy = energyGrid[gridIndex];
                    for (n = 0; n < numAtoms; n++) {
                        float dx = x - atoms[n].x;
                        float dy = y - atoms[n].y;
                        float dz = z - atoms[n].z;
                        energy += atoms[n].charge / sqrtf(dx * dx + dy * dy + dz * dz);
                    }
                    // add old and new energy values and store them
                    energyGrid[gridIndex] = energy + oldEnergy;
                    __syncthreads();

                }
        }
    }
}

/*
 * d_discombobulateKernelConst
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
    // check to ensure thread is supposed to be doing work
    if (blockDim.x * blockIdx.x + threadIdx.x < grid.x) {
        float x = gridSpacing * (threadIdx.x + blockIdx.x * blockDim.x);

            for (i = 0; i < grid.z; i++) {
                float z = gridSpacing * (float) i;
                for (j = 0; j < grid.y; j++) {
                    float y = gridSpacing * (float) j;
                    int gridIndex = grid.x * grid.y * i + grid.x * j + (blockIdx.x * blockDim.x + threadIdx.x);
                    float energy = 0.0f;
                    // load early to offset loading time before use
                    float oldEnergy = energyGrid[gridIndex];
                    for (n = 0; n < numAtoms; n++) {
                        float dx = x - constAtoms[n].x;
                        float dy = y - constAtoms[n].y;
                        float dz = z - constAtoms[n].z;
                        float charge = constAtoms[n].charge;
                        energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
                    }
                    // add old and new energy values and store them
                    energyGrid[gridIndex] = energy + oldEnergy;
                    __syncthreads();

                }
            }
    }
}

/*
 * d_discombobulateKernelConst2D
 * A kernel that calculates the electrostatic potential and stores it in a float array,
    this version uses constant memory instead of passing in an atoms array, Parallelizes
    on the x and y Axis
 *
 * energyGrid: A float array of the sample points in and around the molecule
 * grid: a dim3 struct containing the dimensions of energyGrid
 * gridSpacing: the space between grid points along each axis
 *   numAtoms: number of atoms in constAtoms
 */

__global__ void d_discombulateKernelConst2D(float * energyGrid, dim3 grid, float gridSpacing,
                                          int numAtoms) {


    int i, n;
    // computes indexes in x and y axis from block and thread
    int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockDim.y * blockIdx.y + threadIdx.y;
    // check to ensure thread is supposed to be doing work
    if (idX < grid.x && idY < grid.y) {
        float x = gridSpacing * (float) idX;
        float y = gridSpacing * (float) idY;
        for (i = 0; i < grid.z; i++) {
            float z = gridSpacing * (float) i;
            int gridIndex = grid.x * grid.y * i + grid.x * idY + idX;
            float energy = 0.0f;
            // load early to offset loading time before use
            float oldEnergy = energyGrid[gridIndex];
            for (n = 0; n < numAtoms; n++) {
                float dx = x - constAtoms[n].x;
                float dy = y - constAtoms[n].y;
                float dz = z - constAtoms[n].z;
                float charge = constAtoms[n].charge;
                energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
            }
            // add old and new energy values and store them
            energyGrid[gridIndex] = energy + oldEnergy;
            __syncthreads();

        }
    }
}


__global__ void d_discombulateKernelConst3D(float * energyGrid, dim3 grid, float gridSpacing,
                                            int numAtoms) {


    int n;
    // computes indexes in x, y, and z axis from block and thread
    int idX = blockDim.x * blockIdx.x + threadIdx.x;
    int idY = blockDim.y * blockIdx.y + threadIdx.y;
    int idZ = blockDim.z * blockIdx.z + threadIdx.z;

    // check to ensure thread is supposed to be doing work
    if (idX < grid.x && idY < grid.y && idZ < grid.z) {
        float x = gridSpacing * (float) idX;
        float y = gridSpacing * (float) idY;
        float z = gridSpacing * (float) idZ;
        int gridIndex = grid.x * grid.y * idZ + grid.x * idY + idX;
        float energy = 0.0f;
        // load early to offset loading time before use
        float oldEnergy = energyGrid[gridIndex];
        for (n = 0; n < numAtoms; n++) {
            float dx = x - constAtoms[n].x;
            float dy = y - constAtoms[n].y;
            float dz = z - constAtoms[n].z;
            float charge = constAtoms[n].charge;
            energy += charge / sqrtf(dx * dx + dy * dy + dz * dz);
        }
        // add old and new energy values and store them
        energyGrid[gridIndex] += energy + oldEnergy;
        __syncthreads();


    }
}




// an empty kernel to improve timing?
__global__ void emptyKernel()
{
}