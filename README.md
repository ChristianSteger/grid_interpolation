# Description
Fortran code to interpolate from an unstructured grid (triangle mesh) to a regular grid and vice versa.
The former interpolation is performed with inverse distance weighting (IDW) for an unstructured grid 
or barycentric interpolation for a triangle mesh. The latter interpolation is performed bilinearly.
The nearest neighbours for the IDW interpolation can be found with a k-d tree or a more efficient algorithm 
in case of an equally spaced regular grid (esrg) &ndash; i.e., a grid with an equal spacing in the x- and y-direction.
For the barycentric interpolation, triangles are found with a 'triangle walk'-algorithm.

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

## Example application
![Alt text](https://github.com/ChristianSteger/Media/blob/master/grid_interpolation/Re-interpolation.png?raw=true "Output from test_interpolation.py")
The above image visualises the application of the interpolation routines. We start with 5000 randomly positioned
points whose z-values reflect a 'sine-mountain-pattern'. The points are subsequently Delaunay triangulated and its node values interpolated to the centres of a regular grid (via barycentric interpolation). An extrapolation is performed for
cell centres of the regular grid that are located outside of the convex hull. Finally, the z-values on the regular
grid are re-interpolated to the points via bilinear interpolation. Larger deviations occur at the boundary
of the domain.


## To do
- 'Lists' of points (*points_cons* and *points_cons_next*) in *query_esrg.f90* are currently set to a fixed size of 1000. For larger numbers, a memory out of bound error occurs. This issue could be resolved by implementing an array with a dynamic size, similar to a C++ vector (see e.g., https://stackoverflow.com/questions/8384406/how-to-increase-array-size-on-the-fly-in-fortran).
- Barycentric interpolation generates larger errors near the edge of the domain (convex hull). These errors could be
reduced by prohibiting barycentric interpolation for sliver triangles close to the edge (and filling the subsequent
missing values from the nearest neighbours).

## References
- Triangle walk (visibility walk): https://inria.hal.science/inria-00072509/document
- Barycentric interpolation: https://codeplea.com/triangular-interpolation