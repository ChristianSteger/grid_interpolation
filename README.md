# Description
Fortran code to interpolate from regular to unstructured grid (e.g., triangle mesh) and vice versa.
The former interpolation is performed with bilinear interpolation, the latter with inverse distance weighting (IDW)
and the application of a k-d tree to efficiently find the nearest neighbours.
For an equally spaced regular grid (esrg) &ndash; i.e., a grid with an equal spacing in the x- and y-direction &ndash;
an even faster algorithm for IDW interpolation is implemented.

# Usage

## Create conda environment
```bash
conda create -n grid_interpolation numpy scipy matplotlib ipython gfortran meson openmp -c conda-forge
```

## Compile Fortran code and build shared library for F2PY (tested on MacOS)
```bash
gfortran -shared -O3 -o libkd_tree.so -fPIC kd_tree.f90
gfortran -shared -O3 -o libquery_esrg.so -fPIC query_esrg.f90
gfortran -shared -o libtriangle_walk.so -fPIC triangle_walk.f90
cwd=$(pwd)
f2py -c --f90flags='-fopenmp' -lgomp --fcompiler=gfortran -L${cwd}/ -I${cwd}/ -lkd_tree -lquery_esrg -m interpolation interpolation.f90
```

## Clean build
```bash
rm *.so *.mod
```

## To do
- 'Lists' of points (*points_cons* and *points_cons_next*) in *idw_interp_esrg.f90* are currently set to a fixed size of 1000. For larger numbers, a memory out of bound error ocurrs. This issue could be resolved by implementing an array with a dynamic size, similar to a C++ vector (see e.g., https://stackoverflow.com/questions/8384406/how-to-increase-array-size-on-the-fly-in-fortran).