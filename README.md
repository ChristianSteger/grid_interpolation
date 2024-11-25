# Description
Fortran code to interpolate from regular to unstructured grid (triangle mesh) and vice versa.
The former interpolation is performed with bilinear interpolation. The latter either with
inverse distance weighting (IDW) for an unstructured grid or barycentric interpolation for
a triangle mesh. The nearest neighbours for the IDW interpolation can be found with a k-d tree
or a specific more efficient algorithm in case of an equally spaced regular grid (esrg) &ndash; i.e., a grid with an equal spacing in the x- and y-direction.

# Usage

## Create conda environment
```bash
conda create -n grid_interpolation numpy scipy matplotlib ipython gfortran meson openmp shapely -c conda-forge
```

## Compile Fortran code and build shared library for F2PY (tested on MacOS)
```bash
gfortran -shared -O3 -o libkd_tree.so -fPIC kd_tree.f90
gfortran -shared -O3 -o libquery_esrg.so -fPIC query_esrg.f90
gfortran -shared -o libtriangle_walk.so -fPIC triangle_walk.f90
cwd=$(pwd)
f2py -c --f90flags='-fopenmp' -lgomp --fcompiler=gfortran -L${cwd}/ -I${cwd}/ -lkd_tree -lquery_esrg -ltriangle_walk -m interpolation interpolation.f90
```

## Clean build
```bash
rm *.so *.mod
```

## To do
- 'Lists' of points (*points_cons* and *points_cons_next*) in *query_esrg.f90* are currently set to a fixed size of 1000. For larger numbers, a memory out of bound error ocurrs. This issue could be resolved by implementing an array with a dynamic size, similar to a C++ vector (see e.g., https://stackoverflow.com/questions/8384406/how-to-increase-array-size-on-the-fly-in-fortran).