# bobTheBuildersMolecularDiscombobulator
[![C/C++ CI](https://github.com/crispyman/bobTheBuildersMolecularDiscombobulator/actions/workflows/c-cpp.yml/badge.svg)](https://github.com/crispyman/bobTheBuildersMolecularDiscombobulator/actions/workflows/c-cpp.yml)

bobTheBuildersMolecularDiscombobulator takes in the atomic structure of a molecule and returns a Flattened 3D map of electrostatic fields in and around the molecule as a CSV.

**To build and run:**
``` bash
git clone https://github.com/crispyman/bobTheBuildersMolecularDiscombobulator.git

make

./main ./stripped_alinin.pqr
```

**To preprocess .pqr files on Linux:**

cat h2o.pqr | tr -s ' ' > h2o2.pqr