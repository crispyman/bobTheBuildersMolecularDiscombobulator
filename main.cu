//
// Created by andrewiii on 5/9/22.
//

#include <string.h>
#include <typeinfo>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include "molecule.h"
#include "csvparser.h"
#include "main.h"
#include "d_main.h"
#include "h_main.h"
#include "config.h"



int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);
int checkGrid(float *ref, float *check, int gridLength);

void printAtoms(atom * atoms, int numAtoms);

int main(int argc, char * argv[])
{
    // Get the file name and parse it.
    char delim = ' ';
    int numAtoms = 0;

    char* file = "stripped_alinin.pqr";
    CsvParser * csvParser = CsvParser_new(file, &delim, 0);
    // Read the molecule file and write the atoms to an array of atoms.
    atom * atoms = readMolecule(csvParser, &numAtoms);
    // Print the whole list of atoms.
    // printAtoms(atoms, numAtoms);


    CsvParser_destroy(csvParser);
    // Allocate the molecule array.
    float * molecule = (float *) malloc(sizeof(float) * 4 * numAtoms);
    float maxX = 0;
    float maxY = 0;
    float maxZ = 0;

    float minX = 0;
    float minY = 0;
    float minZ = 0;

    for (int i = 0; i < numAtoms; i++){
        printf("%s, %f, %f, %f, %f\n", atoms[i].name,
                                    atoms[i].x,
                                    atoms[i].y,
                                    atoms[i].z,
                                    atoms[i].charge);
        molecule[i * 4] = atoms[i].x;
        if (atoms[i].x > maxX)
            maxX = atoms[i].x;
        else if (atoms[i].x > maxX)
            minX = atoms[i].x;

        molecule[i * 4 + 1] = atoms[i].y;
        if (atoms[i].y > maxY)
            maxY = atoms[i].y;
        else if (atoms[i].y > maxY)
            minY = atoms[i].y;

        molecule[i * 4 + 2] = atoms[i].z;
        if (atoms[i].z > maxZ)
            maxZ = atoms[i].z;
        else if (atoms[i].z > maxZ)
            minZ = atoms[i].z;

//        if (atoms[i].name[0] == 'H')
//            molecule[i * 4 + 3] = 1.0;
//        else if (atoms[i].name[0] == 'O')
//            molecule[i * 4 + 3] = -2.0;
    }

    int dimX  = (int) ((abs(maxX) + PADDING) + (int) (abs(minX) + PADDING)) * (1/GRIDSPACING);
    int dimY  = (int) ((abs(maxY) + PADDING) + (int) (abs(minY) + PADDING)) * (1/GRIDSPACING);
    int dimZ = (int) ((abs(maxZ) + PADDING) + (int) (abs(minZ) + PADDING))* (1/GRIDSPACING);

    float * energyGrid_cpu = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    assert(energyGrid_cpu);
    printf("%d * %d * %d = %d\n",dimX, dimY, dimZ, dimX * dimY * dimZ);

    float h_time = discombob_on_cpu(energyGrid_cpu, molecule, dimX, dimY, dimZ, GRIDSPACING, numAtoms);

    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);

    float * energyGrid_gpu = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    assert(energyGrid_gpu);


    float d_time = d_discombobulate(energyGrid_gpu, molecule, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 0);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ);
    printf("GPU (0): \t\t%f msec\n", d_time);
    float speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);

    float * energyGrid_gpu_const = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    assert(energyGrid_gpu);

    float d_time_const = d_discombobulate(energyGrid_gpu_const, molecule, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 1);

    checkGrid(energyGrid_cpu, energyGrid_gpu_const, dimX * dimY * dimZ);
    printf("GPU (1): \t\t%f msec\n", d_time_const);
    speedup = h_time/d_time_const;
    printf("Speedup: \t\t\t%f\n", speedup);


    free(atoms);
    free(molecule);
    free(energyGrid_cpu);
    free(energyGrid_gpu);

}

/*
    getMoleculeLength
    Assigns the number of atoms in the molecule to atomCount

    csvParser: A CsvParser that is parsing the file  that describes the molecule.
    atomCount: The number of atoms in the molecule.
*/
void getMoleculeLength(CsvParser * csvParser, int * atomCount) {
    // Make a copy of the csv parser.
    int count = 0;
    char delim = csvParser->delimiter_;
    char * file = csvParser->filePath_;
    CsvParser * countParser = CsvParser_new(file, &delim, 0);

    CsvRow * row = CsvParser_getRow(countParser);
    while (strcmp(row->fields_[0], "END") != 0) {
        if (strcmp(row->fields_[0], "ATOM") == 0) {
            count++;
        }
        row = CsvParser_getRow(countParser);
    }
    *atomCount = count;
}

/*
    readMolecule
    Reads the molecule file in the parser, parses it and makes an array of atoms.

    csvParser: A parser to parse the CSV file that contains data about the molecule.
    atomCnt: A variable to return the number of atoms in the molecule.
*/
atom * readMolecule(CsvParser * csvParser, int* atomCnt) {
    // Get the number of atoms in the molecule.
    getMoleculeLength(csvParser, atomCnt);
    // Allocate an array of atoms.
    atom * atoms = (atom *) calloc(*atomCnt, sizeof(atom));
    // Get the first row of the file.
    CsvRow * csvRow = CsvParser_getRow(csvParser);
    // delete the row because we don't need it
    CsvParser_destroy_row(csvRow);

    // printf("Number of atoms: %d", *atomCnt);
    // Loop through all lines in the file until the END record.

    for (int i = 0; i < *atomCnt; i++){
        // Skip any record that is not an atom.
        csvRow = CsvParser_getRow(csvParser);
        // printf("Name CSV: %s\nx-coord: %f\ny-coord: %f\nz-coord: %f\ncharge: %f\n", csvRow->fields_[2],
        //                                                                             strtof(csvRow->fields_[5], NULL),
        //                                                                             strtof(csvRow->fields_[6], NULL),
        //                                                                             strtof(csvRow->fields_[7], NULL),
        //                                                                              strtof(csvRow->fields_[8], NULL));

        if (strcmp(*CsvParser_getFields(csvRow), "ATOM") == 0) {

            strcpy(atoms[i].name, csvRow->fields_[2]);
            atoms[i].x = strtof(csvRow->fields_[5], NULL);
            atoms[i].y = strtof(csvRow->fields_[6], NULL);
            atoms[i].z = strtof(csvRow->fields_[7], NULL);
            atoms[i].charge = strtof(csvRow->fields_[8], NULL);
            CsvParser_destroy_row(csvRow);

        }
        // printf("Name: %s\nx-coord: %f\ny-coord: %f\nz-coord: %f\ncharge: %f\n", atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z, atoms[i].charge);
    }

    return atoms;
}


/*
    Prints an atom.
*/
void printAtoms(atom * atoms, int numAtoms) {
    for ( int i = 0; i < numAtoms; i++) {
        printf("Name: %s, \n", atoms[i].name);
        printf("X: %f, \n", atoms[i].x);
        printf("Y: %f, \n", atoms[i].y);
        printf("Z: %f, \n", atoms[i].z);
        printf("Charge: %f, \n", atoms[i].charge);
    }
}


int checkGrid(float *ref, float *check, int gridLength) {
    float*correct = (float *) ref;
    float*output = (float *) check;
    for (int i = 0; i < gridLength; i++) {
        if (output[i] != correct[i]) {
            printf("Incorrect value at [%d]\n", i);
            printf("%f != %f\n", output[i], correct[i]);

            //unixError(errorMsg);
            return 1;
        }
    }

    printf("image is correct\n");
    return 0;
}