//
// Created by andrewiii on 5/10/22.
//
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "CHECK.h"
#include "molecule.h"
#include "h_main.h"

void discombob(double * energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing, int numAtoms);


int discombob_on_cpu(double * energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing, int numAtoms){

    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;

    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    CHECK(cudaEventRecord(start_cpu));

    discombob(energyGrid, atoms, dimX, dimY, dimZ, gridSpacing, numAtoms);

    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

void discombob(double * energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing, int numAtoms) {
    int i,j,k,n;
    for (k=0; k<dimZ; k++) {
        float z = gridSpacing * (float)k;
        for (j = 0; j < dimY; j++) {
            float y = gridSpacing * (float)j;
            for (i = 0; i < dimX; i++){
                float x = gridSpacing * (float)i;
                double energy = 0.0;
                for (n = 0; n<numAtoms; n++){
                    double dx = (double)x - atoms[n].x;
                    double dy = (double)y - atoms[n].y;
                    double dz = (double)z - atoms[n].z;
                    double charge = atoms[n].charge;
                    energy += charge/sqrt(dx*dx + dy*dy + dz*dz);
                }
                energyGrid[dimX*dimY*k + dimX*j + i] = energy;
            }
        }
    }
}
