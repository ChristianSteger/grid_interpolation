# Description
Fortran code to interpolate from regular to unstructured grid (e.g. triangle mesh) and vice versa.
The former interpolation is performed with bilinear interpolation, the latter with inverse distance weighting (IDW) and the application of a k-d tree to efficiently find the nearest neighobours.

# Usage

## Create conda environment
```bash
conda create -n grid_interpolation numpy scipy matplotlib ipython gfortran meson -c conda-forge
```

## Compile Fortran code and build shared library for F2PY
```bash
gfortran -shared -O3 -o libkd_tree.so -fPIC kd_tree.f90
cwd=$(pwd)
f2py -c --fcompiler=gfortran -L${cwd}/ -I${cwd}/ -lkd_tree -m interpolation interpolation.f90
```

## Clean build
```bash
rm *.so kd_tree.mod
```