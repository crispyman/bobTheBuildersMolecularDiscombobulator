//
// Created by andrewiii on 5/10/22.
//
#define MAXFILENAMELEN 32

int fequal(float a, float b);
int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);
int checkGrid(float *ref, float *check, int gridLength, const char* kernelName);
void printAtoms(atom * atoms, int numAtoms);
void writeGrid(float * data, int gridLength, const char* fileName);
