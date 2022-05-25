/* 
    A config file for bobTheBuildersDiscombobulator 
*/
// The amount to pad the energy grid in each direction.
#define PADDING 1
// Granularity of energy calculations
#define GRIDSPACING .1
#define THREADSPERBLOCK 32
#define THREADSPERBLOCK2D 16
#define THREADSPERBLOCK3D 8
#define MAXCONSTANTATOMS 20 // The max number of atoms to copy to constant memory on the device. 

#define PRECISIONTHRESH .01