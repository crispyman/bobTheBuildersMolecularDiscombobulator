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
__constant__ atom constAtoms[MAXCONSTANTATOMS];

int d_discombobulate(float * energyGrid, atom * atoms, int dimX, int dimY, int dimZ, float gridSpacing,  int numAtoms, int which){
    cudaEvent_t start_gpu, stop_gpu;
    float gpuMsecTime = -1;

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

    // Selects which kernel to launch. 
    if (which == 0) {
        atom * d_atoms;         // The array of atoms for the device. 
        CHECK(cudaMalloc((void**)&d_atoms, numAtoms * sizeof(atom)));
        CHECK(cudaMemcpy(d_atoms, atoms, numAtoms * sizeof(atom), cudaMemcpyHostToDevice));

        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);
        d_discombulateKernel<<<gridDim, blockDim>>>(d_energyGrid, d_atoms, grid, gridSpacing, numAtoms);

        CHECK(cudaFree(d_atoms));
    }
    // Same kernel as previous, but this time using constant memory for the atoms array.
    else if (which == 1) {
        dim3 blockDim(THREADSPERBLOCK, 1, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK), 1, 1);

        // Splitting the atoms array into sections to reduce the number of atoms in constant memory
        int numAtomsRemaining = numAtoms;
        for (int i = 0; i < numAtoms/MAXCONSTANTATOMS; i++){
            // Copy atoms to constant memory on the device.
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[i * MAXCONSTANTATOMS], sizeof(atom) * MAXCONSTANTATOMS));
            d_discombulateKernelConst<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, MAXCONSTANTATOMS);
            numAtomsRemaining -= MAXCONSTANTATOMS;

        }
        if (numAtomsRemaining < MAXCONSTANTATOMS) {
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[numAtoms - numAtomsRemaining], sizeof(atom) * numAtomsRemaining));
            d_discombulateKernelConst<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtomsRemaining);
        }
    }
    // Using a 2D kernel. 
    else if (which == 2) {
        // Define kernel dimensions outside of loop
        dim3 blockDim(THREADSPERBLOCK2D, THREADSPERBLOCK2D, 1);
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK2D), ceil((1.0 * dimY) / THREADSPERBLOCK2D), 1);
        // Break the atoms array into smaller parts to allow for larger atom lists.
        int numAtomsRemaining = numAtoms;
        for (int i = 0; i < numAtoms/MAXCONSTANTATOMS; i++){
            // Copy atoms to constant memory on the device.
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[i * MAXCONSTANTATOMS], sizeof(atom) * MAXCONSTANTATOMS));
            d_discombulateKernelConst2D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, MAXCONSTANTATOMS);
            numAtomsRemaining -= MAXCONSTANTATOMS;

        }
        if (numAtomsRemaining < MAXCONSTANTATOMS) {
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[numAtoms - numAtomsRemaining], sizeof(atom) * numAtomsRemaining));
            d_discombulateKernelConst2D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtomsRemaining);
        }
    }
    // Using 3D Kernel
    else if (which == 3) {
        // Define the dimensions of the kernel
        dim3 blockDim(THREADSPERBLOCK3D, THREADSPERBLOCK3D, THREADSPERBLOCK3D);
        // Number of blocks in each direction (x, y, z) is the dimension of the block in that direction/THREADSPERBLOCK3D.
        dim3 gridDim(ceil((1.0 * dimX) / THREADSPERBLOCK3D), ceil((1.0 * dimY) / THREADSPERBLOCK3D), ceil((1.0 * dimZ) / THREADSPERBLOCK3D));
                // Break the atoms array into smaller parts to allow for larger atom lists.
        // Break the atoms array into smaller parts to allow for larger atom lists.
        int numAtomsRemaining = numAtoms;
        for (int i = 0; i < numAtoms/MAXCONSTANTATOMS; i++){
            // Copy atoms to constant memory on the device.
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[i * MAXCONSTANTATOMS], sizeof(atom) * MAXCONSTANTATOMS));
            d_discombulateKernelConst3D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, MAXCONSTANTATOMS);
            numAtomsRemaining -= MAXCONSTANTATOMS;

        }
        if (numAtomsRemaining < MAXCONSTANTATOMS) {
            CHECK(cudaMemcpyToSymbol(constAtoms, &atoms[numAtoms - numAtomsRemaining], sizeof(atom) * numAtomsRemaining));
            d_discombulateKernelConst3D<<<gridDim, blockDim>>>(d_energyGrid, grid, gridSpacing, numAtomsRemaining);
        }
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
                        float charge = atoms[n].charge;
                        energy += charge/sqrt(dx*dx + dy*dy + dz*dz);

                    }
                    // Write the resulting energy back to the grid index.
                    energyGrid[gridIndex] = energy + oldEnergy; 
                    __syncthreads();
            }
        }
    }
}

/*
 * d_discombobulateKernelConst
 * A kernel that calculates the electrostatic potential and stores it in a float array,
 * this version uses constant memory instead of passing in an atoms array
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
                        energy += charge/sqrt(dx*dx + dy*dy + dz*dz);

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

/* 
    OVERALL SUMMARY: 
    Basically the same kernel as the previous constant kernel, but in this case, each thread handles
    only one (x,y) and every z associated with that (x,y). 
*/

    int i, n;                                                   // Iterator
    int idX = blockDim.x * blockIdx.x + threadIdx.x;            // Thread x index 
    int idY = blockDim.y * blockIdx.y + threadIdx.y;            // Thread y index
    if (idX < grid.x && idY < grid.y) {                         // If thread index is on the grid in both x and y direcion
        float x = gridSpacing * (float) idX;                    // X-index on the grid for the current thread.
        float y = gridSpacing * (float) idY;                    // Y-index on the grid for the current thread.
        // For each z-index into the grid
        for (i = 0; i < grid.z; i++) {                          // The z-index of the current slice. 
            // Calculate the grid index
            int gridIndex = grid.x * grid.y * i + grid.x * idY + idX;
            float z = gridSpacing * (float) i;                  
            float energy = 0.0f;
            // load early to offset loading time before use
            float oldEnergy = energyGrid[gridIndex];            // 
            for (n = 0; n < numAtoms; n++) {
                float dx = x - constAtoms[n].x;
                float dy = y - constAtoms[n].y;
                float dz = z - constAtoms[n].z;
                float charge = constAtoms[n].charge;
                energy += charge/sqrt(dx*dx + dy*dy + dz*dz);

            }
            // add old and new energy values and store them
            energyGrid[gridIndex] = energy + oldEnergy;
            __syncthreads();
        }
    }
}


__global__ void d_discombulateKernelConst3D(float * energyGrid, dim3 grid, float gridSpacing,
                                            int numAtoms) {

/* 
    OVERALL SUMMARY: 
    Basically the same kernel as the previous constant kernel, but in this case, each thread handles
    only one (x,y,z).
*/

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
            energy += charge/sqrt(dx*dx + dy*dy + dz*dz);

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