//
// Created by andrewiii on 5/9/22.
//

#include <string.h>
#include <typeinfo>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include "csvparser.h"
#include "main.h"
#include "d_main.h"
#include "h_main.h"



int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);
const int padding = 10;

int main(int argc, char * argv[])
{
    // Get the file name and parse it. 
    char* file = "h2o2.xyz";
    char delim = ' ';
    // Make the parser giving the correct delimeter.
    CsvParser * csvParser = CsvParser_new(file, &delim, 0);
    int numAtoms;
    atom * atoms = readMolecule(csvParser, &numAtoms);
    CsvParser_destroy(csvParser);
    float * molecule = (float *) malloc(sizeof(float) * 3 * numAtoms);
    float maxX = 0;
    float maxY = 0;
    float maxZ = 0;

    float minX = 0;
    float minY = 0;
    float minZ = 0;

    for (int i = 0; i < numAtoms; i++){
        printf("%s, %f, %f, %f\n", atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
        molecule[i * 3] = atoms[i].x;
        if (atoms[i].x > maxX)
            maxX = atoms[i].x;
        else if (atoms[i].x > maxX)
            minX = atoms[i].x;

        molecule[i * 3 + 1] = atoms[i].y;
        if (atoms[i].y > maxY)
            maxY = atoms[i].y;
        else if (atoms[i].y > maxY)
            minY = atoms[i].y;

        molecule[i * 3 + 2] = atoms[i].z;
        if (atoms[i].z > maxZ)
            maxZ = atoms[i].z;
        else if (atoms[i].z > maxZ)
            minZ = atoms[i].z;
    }
    int dimX  = (int) (abs(maxX) + padding) + (int) (abs(minX) + padding);
    int dimY  = (int) (abs(maxY) + padding) + (int) (abs(minY) + padding);
    int dimZ = (int) (abs(maxZ) + padding) + (int) (abs(minZ) + padding);

    float * energyGrid = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    float gridSpacing = 0.001;

    discombob_on_cpu(energyGrid, dimX, dimY, dimZ, gridSpacing, molecule, numAtoms);



    free(atoms);
    free(molecule);
    free(energyGrid);
}


int getMoleculeLength(CsvRow * csvRow) {
    const char **rowFields = CsvParser_getFields(csvRow);
    return strtol(rowFields[0], NULL, 10);
}

atom * readMolecule(CsvParser * csvParser, int* atomCnt) {
    CsvRow *csvRow = CsvParser_getRow(csvParser);
    int numAtoms = getMoleculeLength(csvRow);
    *atomCnt = numAtoms;
    CsvParser_destroy_row(csvRow);

    csvRow = CsvParser_getRow(csvParser);
    CsvParser_destroy_row(csvRow);

    atom *atoms = (atom *) calloc(numAtoms, sizeof(atom));
    for (int j = 0; j < numAtoms; j++) {
        csvRow = CsvParser_getRow(csvParser);
        const char **rowFields = CsvParser_getFields(csvRow);
        //if (CsvParser_getNumFields(csvRow) != 4 || rowFields[0][0] < 'A' || rowFields[0][0] > 'Z') {
        //    return NULL;
        //}
        assert(CsvParser_getNumFields(csvRow) == 4);
        strcpy(atoms[j].name, rowFields[0]);
        atoms[j].x = strtof(rowFields[1], NULL);
        atoms[j].y = strtof(rowFields[2], NULL);
        atoms[j].z = strtof(rowFields[3], NULL);
        CsvParser_destroy_row(csvRow);

    }
    return atoms;

}