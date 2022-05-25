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
#include "csvwriter.h"
#include "main.h"
#include "d_main.h"
#include "h_main.h"
#include "config.h"



int main(int argc, char * argv[])
{
    // Get the file name and parse it.
    char delim = ' ';
    int numAtoms = 0;
    const char* file = "stripped_alinin.pqr";
    CsvParser * csvParser = CsvParser_new(file, &delim, 0);
    // Read the molecule file and write the atoms to an array of atoms.
    atom * atoms = readMolecule(csvParser, &numAtoms);

    CsvParser_destroy(csvParser);

    // Get the maximum and minimum coordinates in all 3 directions for any atom.
    float maxX = 0;
    float maxY = 0;
    float maxZ = 0;
    float minX = 0;
    float minY = 0;
    float minZ = 0;
    for (int i = 0; i < numAtoms; i++){
        if (atoms[i].x > maxX)
            maxX = atoms[i].x;
        else if (atoms[i].x > maxX)
            minX = atoms[i].x;

        if (atoms[i].y > maxY)
            maxY = atoms[i].y;
        else if (atoms[i].y > maxY)
            minY = atoms[i].y;

        if (atoms[i].z > maxZ)
            maxZ = atoms[i].z;
        else if (atoms[i].z > maxZ)
            minZ = atoms[i].z;
    }

    int dimX  = (int) ((abs(maxX) + PADDING) + (int) (abs(minX) + PADDING)) * (1/GRIDSPACING);
    int dimY  = (int) ((abs(maxY) + PADDING) + (int) (abs(minY) + PADDING)) * (1/GRIDSPACING);
    int dimZ = (int) ((abs(maxZ) + PADDING) + (int) (abs(minZ) + PADDING))* (1/GRIDSPACING);

    // Shift the coordinates of all atoms to be positive plus padding.
    for (int i = 0; i < numAtoms; i++) {
        atoms[i].x  += (abs(minX) + PADDING);
        atoms[i].y += (abs(minY) + PADDING);
        atoms[i].z += (abs(minZ) + PADDING);
     }

    // CPU
    float * energyGrid_cpu = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    assert(energyGrid_cpu);
    float h_time = discombob_on_cpu(energyGrid_cpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms);
    writeGrid(energyGrid_cpu, dimX * dimY * dimZ, "cpu.csv");

    printf("\nTiming\n");
    printf("------\n");
    printf("CPU: \t\t\t\t%f msec\n", h_time);

    // GPU
    float * energyGrid_gpu = (float *) malloc(sizeof(float) * dimX * dimY * dimZ);
    assert(energyGrid_gpu);

    float d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 0);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ, "Simple Kernel");
    printf("GPU (0): \t\t%f msec\n", d_time);
    float speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);
    writeGrid(energyGrid_gpu, dimX * dimY * dimZ, "gpusimple.csv");


    // GPU Const
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(float) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 1);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ, "1D Const Kernel");
    printf("GPU(1): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);
    writeGrid(energyGrid_gpu, dimX * dimY * dimZ, "gpu1D.csv");


    // GPU Const 2D
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(float) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 2);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ, "2D Const Kernel");
    printf("GPU (2): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);
    writeGrid(energyGrid_gpu, dimX * dimY * dimZ, "gpu2D.csv");


    // GPU Const 3D
    d_time = 0;
    memset(energyGrid_gpu, 0 , sizeof(float) * dimX * dimY * dimZ);

    d_time = d_discombobulate(energyGrid_gpu, atoms, dimX, dimY, dimZ, GRIDSPACING, numAtoms, 3);

    checkGrid(energyGrid_cpu, energyGrid_gpu, dimX * dimY * dimZ, "3D Const Kernel");
    printf("GPU (3): \t\t%f msec\n", d_time);
    speedup = h_time/d_time;
    printf("Speedup: \t\t\t%f\n", speedup);
    writeGrid(energyGrid_gpu, dimX * dimY * dimZ, "gpu3D.csv");


    free(atoms); 
    //free(molecule);

    free(energyGrid_cpu);
    free(energyGrid_gpu);


}

/*
    getMoleculeLength
    Assigns the number of atoms in the molecule to atomCount

    filepath: A path to the pqr file describing the molecule

    returns: 
        count: A count of the number of 'ATOM' records in the pqr file.

*/
int getMoleculeLength(char * filepath) {
    // Open the file. 
    char str[20];
    FILE *fptr;
    fptr = fopen(filepath, "r");


    char *pos;
    int index, count;
    
    count = 0;
    // Read from the file until we reach the end.
    while ((fgets(str, 20, fptr)) != NULL)
    {
        index = 0;
        // While strstr doesnt return NULL.
        // Index and pos are needed to make so this is not an
        // infinite loop. strstr simply returns the 
        while ((pos = strstr(str + index, "ATOM")) != NULL)
        {
            // Update the current index to be the location
            // ATOM was found and increment by 1 to avoid 
            // recounting it. 
            index = (pos - str) + 1;
            count++;
        }
    }
    return count;
} 

/*
    readMolecule
    Reads the molecule file in the parser, parses it and makes an array of atoms.

    csvParser: A parser to parse the CSV file that contains data about the molecule.
    atomCnt: A variable to return the number of atoms in the molecule.
*/
atom * readMolecule(CsvParser * csvParser, int* atomCnt) {
    // Get the number of atoms in the molecule.
    *atomCnt = getMoleculeLength(csvParser->filePath_);
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
        if (strcmp(*CsvParser_getFields(csvRow), "ATOM") == 0) {
            atoms[i].x = strtof(csvRow->fields_[5], NULL);
            atoms[i].y = strtof(csvRow->fields_[6], NULL);
            atoms[i].z = strtof(csvRow->fields_[7], NULL);
            atoms[i].charge = strtof(csvRow->fields_[8], NULL);
            CsvParser_destroy_row(csvRow);
        }
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

/* 
    fequal: Returns 1 if the two floating point values are more different than a threshold.
*/
int fequal(float a, float b) {
    float diff = abs(a - b);
    if ((diff < PRECISIONTHRESH)  || (isinf(a) && isinf(b))) {
        // Equal
        return 0;
    }
    // Not equal.
    return 1;
}

int checkGrid(float *ref, float *check, int gridLength, const char* kernelName) {
    float*correct = (float *) ref;
    float*output = (float *) check;
    for (int i = 0; i < gridLength; i++) {
        if (fequal(correct[i], ref[i])) {
            printf("\e[1;31m%s\e[0m priduced an incorrect value at [%d]\n",kernelName, i);
            printf("Actual: %f != Expected: %f\n", output[i], correct[i]);
            return 1;
        }
    }

    printf("\e[1;32m%s\e[0m produced a correct grid.\n", kernelName);
    return 0;
}


void writeGrid(float * data, int gridLength, const char* fileName){
    char buf[1024];
    float max = 1;
    CsvWriter *csvWriter = CsvWriter_new(fileName, ",", 0);
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
    // printf("\n%f\n", max);
}