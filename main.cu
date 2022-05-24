//
// Created by andrewiii on 5/9/22.
//

#include <string.h>
#include <typeinfo>
#include <ctype.h>
#include <assert.h>
#include <stdlib.h>
#include <limits>
#include "molecule.h"
#include "csvparser.h"
#include "csvwriter.h"
#include "main.h"
#include "d_main.h"
#include "h_main.h"
#include "config.h"



int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);
int checkGrid(double *ref, double *check, int gridLength);
void printAtoms(atom * atoms, int numAtoms);
void writeGrid(double * data, int gridLength);



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
    //float * molecule = (float *) malloc(sizeof(float) * 4 * numAtoms);
    double maxX = 0;
    double maxY = 0;
    double maxZ = 0;

    double minX = 0;
    double minY = 0;
    double minZ = 0;

    for (int i = 0; i < numAtoms; i++){
        // printf("%f, %f, %f, %f\n",// atoms[i].name,
        //                             atoms[i].x,
        //                             atoms[i].y,
        //                             atoms[i].z,
        //                             atoms[i].charge);
        //molecule[i * 4] = atoms[i].x;
        if (atoms[i].x > maxX)
            maxX = atoms[i].x;
        else if (atoms[i].x > maxX)
            minX = atoms[i].x;

        //molecule[i * 4 + 1] = atoms[i].y;
        if (atoms[i].y > maxY)
            maxY = atoms[i].y;
        else if (atoms[i].y > maxY)
            minY = atoms[i].y;

        //molecule[i * 4 + 2] = atoms[i].z;
        if (atoms[i].z > maxZ)
            maxZ = atoms[i].z;
        else if (atoms[i].z > maxZ)
            minZ = atoms[i].z;

        //molecule[i * 4 + 3] = atoms[i].charge;
    }

    int dimX  = (int) ((abs(maxX) + PADDING) + (int) (abs(minX) + PADDING)) * (1/GRIDSPACING);
    int dimY  = (int) ((abs(maxY) + PADDING) + (int) (abs(minY) + PADDING)) * (1/GRIDSPACING);
    int dimZ = (int) ((abs(maxZ) + PADDING) + (int) (abs(minZ) + PADDING))* (1/GRIDSPACING);

    // Normalize positions for padding. 
    for (int i = 0; i < numAtoms; i++) {
        atoms[i].x  += (abs(minX) + PADDING);
        atoms[i].y += (abs(minY) + PADDING);
        atoms[i].z += (abs(minZ) + PADDING);
    }
    // printf("%d * %d * %d * %lu = %lu\n",dimX, dimY, dimZ, sizeof(float), dimX * dimY * dimZ * sizeof(float));

    // CPU
    double * energyGrid_cpu = (double *) malloc(sizeof(double) * dimX * dimY * dimZ);
    assert(energyGrid_cpu);
    float h_time = discombob_on_cpu(energyGrid_cpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms);
    writeGrid(energyGrid_cpu, dimX * dimY * dimZ);


    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);


    // GPU
    double * energyGrid_gpu = (double *) malloc(sizeof(double) * dimX * dimY * dimZ);
    assert(energyGrid_gpu);

    float d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 0);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ);
    printf("GPU (0): \t\t%f msec\n", d_time);
    float speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);


    // GPU Const
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(double) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 1);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ);
    printf("GPU (1): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);



 /*    // GPU Const 2D
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(float) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 2);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ);
    printf("GPU (2): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);

 */
/*     // GPU Const 3D
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(float) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 3);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ);
    printf("GPU (3): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);

 */
    free(atoms);
    //free(molecule);

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

            //strcpy(atoms[i].name, csvRow->fields_[2]);
            atoms[i].x = strtod(csvRow->fields_[5], NULL);
            atoms[i].y = strtod(csvRow->fields_[6], NULL);
            atoms[i].z = strtod(csvRow->fields_[7], NULL);
            atoms[i].charge = strtod(csvRow->fields_[8], NULL);
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
        //printf("Name: %s, \n", atoms[i].name);
        printf("X: %f, \n", atoms[i].x);
        printf("Y: %f, \n", atoms[i].y);
        printf("Z: %f, \n", atoms[i].z);
        printf("Charge: %f, \n", atoms[i].charge);
    }
}

bool fequal(double a, double b)
{
 return fabs(a-b) < __DBL_EPSILON__;
}


int checkGrid(double *ref, double *check, int gridLength) {
    double*correct = (double *) ref;
    double*output = (double *) check;
    for (int i = 0; i < gridLength; i++) {
        if (fequal(output[i], correct[i])) {
            printf("Incorrect value at [%d]\n", i);
            printf("Actual: %f != Expected: %f\n", output[i], correct[i]);

            //unixError(errorMsg);
            return 1;
        }
    }

    printf("Grid is correct\n");
    return 0;
}


void writeGrid(double * data, int gridLength){
    char buf[1024];
    float max = 1;
    CsvWriter *csvWriter = CsvWriter_new("cpuopt.csv", ",", 0);
    for (int i = 0; i < gridLength; i++){
        if (data[i] > max)
            max = data[i];
        gcvt(data[i], 25, buf);
        if (CsvWriter_writeField(csvWriter, buf)) {
            printf("Error: %s\n", CsvWriter_getErrorMessage(csvWriter));
            break;
        }
    }
    CsvWriter_destroy(csvWriter);
    printf("\n%f\n", max);
}