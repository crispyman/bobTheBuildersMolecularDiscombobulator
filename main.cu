//
// Created by andrewiii on 5/9/22.
//

#include <string.h>
#include <typeinfo>
#include <ctype.h>
#include <assert.h>
#include "csvparser.h"
#include "main.h"


int getMoleculeLength(CsvRow * csvRow);
atom * readMolecule(CsvParser * csvParser, int* atomCnt);

int main(int argc, char * argv[])
{

    char* file = "h2o2.xyz";
    char delim = ' ';
    CsvParser * csvParser = CsvParser_new(file, &delim, 0);
    int numAtoms;
    atom * atoms = readMolecule(csvParser, &numAtoms);
    CsvParser_destroy(csvParser);

    for (int i = 0; i < numAtoms; i++){
        printf("%s, %f, %f, %f\n", atoms[i].name, atoms[i].x, atoms[i].y, atoms[i].z);
    }

    free(atoms);
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