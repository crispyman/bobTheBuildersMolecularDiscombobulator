
#include "CHECK.h"
#include "config.h"
#include "molecule.h"
//
// Created by andrewiii on 5/10/22.
//

__global__ void d_discombobulateKernel(float *,int);
/* 
    d_main.cu 
    Calculates an electrostatic potential grid for the molecule passed. 
    
    energyGrid: The grid that will contain the result
    atoms: An array of all atoms of the molecule and their positions. 
    numAtoms: The number of atoms in the molecule.
*/
float * d_discombobulate(energyGrid * energyGrid, float * atoms, int numAtoms) {
    /* 
        TODO: Write host code to set up the cuda kernel, and launch d_discombobulateKernel
    */
}

/* 
    d_discombobulateKernel
    A kernel that calculates the electrostatic potential and stores it 
    in a float array

    energyGrid: The float array associated with the molecule 
*/
__global__ void d_discombulateKernel(float * energyGrid, float gridSpacing, float *atoms, int numAtoms, gridDim dim) {
    /* 
        TODO: Write code to calculate the energy grid and store it in the 
        float array energyGrid. 
    */
}