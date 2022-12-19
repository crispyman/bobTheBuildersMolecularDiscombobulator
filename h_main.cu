#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "CHECK.h"
#include "molecule.h"
#include "h_main.h"

void discombob(float *energyGrid, const atom *atoms, const int dimX, const int dimY, const int dimZ,
               const float gridSpacing, const int numAtoms);


int discombob_on_cpu(float *energyGrid, const atom *atoms, const int dimX, const int dimY, const int dimZ,
                     const float gridSpacing, const int numAtoms) {

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

void discombob(float *energyGrid, const atom *atoms, const int dimX, const int dimY, const int dimZ,
               const float gridSpacing, const int numAtoms) {
    int i, j, k, n;
    for (k = 0; k < dimZ; k++) {
        float z = gridSpacing * (float) k;
        for (j = 0; j < dimY; j++) {
            float y = gridSpacing * (float) j;
            for (i = 0; i < dimX; i++) {
                float x = gridSpacing * (float) i;
                double energy = 0.0f;
                for (n = 0; n < numAtoms; n++) {
                    float dx = x - atoms[n].x;
                    float dy = y - atoms[n].y;
                    float dz = z - atoms[n].z;
                    float charge = atoms[n].charge;
                    energy += charge / sqrt(dx * dx + dy * dy + dz * dz);
                }
                ((float*)energyGrid)[dimX * dimY * k + dimX * j + i] = energy;
            }
        }
    }
}
