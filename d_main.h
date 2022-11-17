int d_discombobulate(float *energyGrid, const atom *atoms, const int dimX, const int dimY, const int dimZ,
                     const float gridSpacing, const int numAtoms, const int which);

int d_discombobulate_multi_GPU(float *energyGrid, const atom *atoms, const int dimX, const int dimY,
                               const int dimZ, const float gridSpacing, const int numAtoms);

int d_discombobulate_multi_GPU_threaded(float *energyGrid, const atom *atoms, const int dimX, const int dimY,
                                        const int dimZ, const float gridSpacing, const int numAtoms);
