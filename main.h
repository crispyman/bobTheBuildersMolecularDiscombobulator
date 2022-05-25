//
// Created by andrewiii on 5/10/22.
//

int fequal(float a, float b);
int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);
int checkGrid(float *ref, float *check, int gridLength);
void printAtoms(atom * atoms, int numAtoms);
void writeGrid(float * data, int gridLength);
