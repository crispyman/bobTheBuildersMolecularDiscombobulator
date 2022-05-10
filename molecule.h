/* 
    molecule.h
    Defines a molecule and its relevent properties. 
*/

typedef struct energyGrid {
    float * energyGrid;
    float gridSpacing;
    int dimX;
    int dimY;
    int dimZ;
} energyGrid;

typedef struct atom {
    char name[3];
    float x;
    float y;
    float z;
} atom;

typedef struct molecule {
    energyGrid * energyGrid;
    atom * atoms;
    int numAtoms; 
} molecule;