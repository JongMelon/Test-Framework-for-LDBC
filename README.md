# Test-Framework-for-LDBC
Based on graph algorithm library of RUC

## Environment

- CUDA 12.2
- CMake 3.9+

## File Structure

- `src/`: source code
- `include/`: header files
- `data/`: put the LDBC dataset here (.properties, .v, .e)

## Build & Run

```shell
mkdir build
cd build
cmake ..
make
./bin/Test
```
