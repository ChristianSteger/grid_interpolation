# Description
Fortran code to interpolate from regular to unstructured grid (e.g., triangle mesh) and vice versa.
The former interpolation is performed with bilinear interpolation, the latter with inverse distance weighting (IDW)
and the application of a k-d tree to efficiently find the nearest neighbours.
For an evenly spaced regular grid (esrg) &ndash; i.e., a grid with an equal spacing in the x- and y-direction &ndash;
an even faster algorithm for IDW interpolation is implemented.

# Usage

## Create conda environment
```bash
conda create -n grid_interpolation numpy scipy matplotlib ipython gfortran meson -c conda-forge
```

## Compile Fortran code and build shared library for F2PY (tested on MacOS)
```bash
gfortran -shared -O3 -o libkd_tree.so -fPIC kd_tree.f90
gfortran -shared -O3 -o libidw_interp_esrg.so -fPIC idw_interp_esrg.f90
cwd=$(pwd)
f2py -c --fcompiler=gfortran -L${cwd}/ -I${cwd}/ -lkd_tree -lidw_interp_esrg -m interpolation interpolation.f90
```

## Clean build
```bash
rm *.so *.mod
```