// The amount to pad the energy grid in each direction.
#define PADDING 4
// Granularity of energy calculations
#define GRIDSPACING .05
#define THREADSPERBLOCK 32
#define THREADSPERBLOCK2D 16
#define THREADSPERBLOCK3D 8
#define MAXCONSTANTATOMS 150 // The max number of atoms to copy to constant memory on the device.

#define NEARZERO 0.000001 // Determines when value is close enough to zero that we can consider it zero
#define ERRORTHRESH 12.0  // acceptable percent difference between CPU and GPU