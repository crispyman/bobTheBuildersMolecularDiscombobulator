//
// Created by andrewiii on 5/10/22.
//

void discombob_on_cpu(float * energyGrid, unsigned int dimX, unsigned int dimY, unsigned int dimZ, float gridSpacing, float *atoms, int numAtoms){
    unsigned int i,j,k,n;
    unsigned int atomArrDim = numAtoms * 4;
    for (k=0; k<dimZ; k++) {
        float z = gridSpacing * (float)k;
        for (j = 0; j < dimY; j++) {
            float y = gridSpacing * (float)j;
            for (i = 0; i<dimX; i++){
                float x = gridSpacing * (float)i;
                float energy = 0.0f;
                for (n = 0; n<atomArrDim; n+=4){
                    float dx = x - atoms[n];
                    float dy = y - atoms[n+1];
                    float dz = z - atoms[n+2];
                    energy += atoms[n+3]/sqrtl(abs((double)dx*dx + dy*dy + dz*dz));
                }
                energyGrid[dimX*dimY*k + dimX*j + i] = energy;
            }
        }
    }
}