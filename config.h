// The amount to pad the energy grid in each direction.
#define PADDING 4
// Granularity of energy calculations
#define GRIDSPACING .5
#define THREADSPERBLOCK 4
#define THREADSPERBLOCK2D_X 8
#define THREADSPERBLOCK2D_Y 8
#define THREADSPERBLOCK3D_X 8
#define THREADSPERBLOCK3D_Y 8
#define THREADSPERBLOCK3D_Z 8
#define COARSEN_FACTOR 4
#define MAXCONSTANTATOMS 150 // The max number of atoms to copy to constant memory on the device.

#define NEARZERO 0.000001 // Determines when value is close enough to zero that we can consider it zero
#define ERRORTHRESH 12.0  // acceptable percent difference between CPU and GPU