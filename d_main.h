int d_discombobulate(float *energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing, int numAtoms,
                     int which);

int d_discombobulate_multi_GPU(float *energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing,
                               int numAtoms);

int d_discombobulate_multi_GPU_threaded(float *energyGrid, atom *atoms, int dimX, int dimY, int dimZ, float gridSpacing,
                                        int numAtoms);
