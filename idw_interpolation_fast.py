# Description: Test interpolation between regular and unstructured grid
# (triangle mesh)
#
# Interpolation/remapping:
# regular -> unstructured grid: bilinear interpolation
# unstructured -> regular grid: inverse distance weighting (IDW)
#
# Author: Christian Steger, October 2024

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from scipy import interpolate
from scipy.spatial import KDTree
from numba import njit, prange

mpl.style.use("classic") # type: ignore

###############################################################################
# Create example data and grid
###############################################################################

# Grid/mesh size
# ----- very small grid/mesh -----
x_size = 10
y_size = 8
num_vertices = 50
# ----- small grid/mesh -----
# x_size = 110
# y_size = 91
# num_vertices = 1_300
# ----- large grid/mesh -----
# x_size = 770
# y_size = 637
# num_vertices = 250_000
# ----------------------

# Regular grid
limits = {"x_min": -50.0, "x_max": 40.0, "y_min": -30.0, "y_max": 40.0}
x_axis = np.linspace(limits["x_min"], limits["x_max"], x_size)
y_axis = np.linspace(limits["y_min"], limits["y_max"], y_size)
print("Number of cells in regular grid: ", (x_axis.size * y_axis.size))

# Example data on regular grid (2-dimensional Gaussian distribution)
x_0 = 0.0
y_0 = 0.0
sigma_x = 25.0
sigma_y = 15.0
A = 25.0
z_reg = A * np.exp(-((x_axis[np.newaxis, :] - x_0) ** 2 / (2 * sigma_x ** 2)) -
                   ((y_axis[:, np.newaxis] - y_0) ** 2 / (2 * sigma_y ** 2)))
#z_reg[:30, :55] = 15.0

# Grid lines
dx = np.diff(x_axis).mean()
dy = np.diff(y_axis).mean()
# dx != dy for fast interpolation method!!!
x_grid = np.hstack((x_axis[0] - dx / 2.0, x_axis[:-1] + np.diff(x_axis) / 2.0,
                    x_axis[-1] + dx / 2.0))
y_grid = np.hstack((y_axis[0] - dy / 2.0, y_axis[:-1] + np.diff(y_axis) / 2.0,
                    y_axis[-1] + dy / 2.0))

# Triangle mesh
vertices = np.empty((num_vertices, 2))
np.random.seed(33)
vertices[:, 0] = np.random.uniform(limits["x_min"], limits["x_max"],
                                   num_vertices)
vertices[:, 1] = np.random.uniform(limits["y_min"], limits["y_max"],
                                   num_vertices)
triangles = Delaunay(vertices)
print("Number of cells in triangle mesh: ", triangles.simplices.shape[0])
# vertices_per_triangle = vertices[triangles.simplices]
# neighbors = triangles.neighbors
# neighbour triangles (-1: boundary -> no triangle)
indptr, indices = triangles.vertex_neighbor_vertices

###############################################################################
# Plot
###############################################################################

# Colormap (absolute)
cmap = plt.get_cmap("Spectral")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(0.0, A)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N) # type: ignore

# Plot
plt.figure()
plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
plt.vlines(x=x_grid, ymin=y_grid[0], ymax=y_grid[-1], colors="black")
plt.hlines(y=y_grid, xmin=x_grid[0], xmax=x_grid[-1], colors="black")
# -----------------------------------------------------------------------------
plt.triplot(vertices[:, 0], vertices[:, 1], triangles.simplices, color="black",
                linewidth=0.5, zorder=2)
plt.scatter(vertices[:, 0], vertices[:, 1], color="black", s=25, zorder=2)
# -----------------------------------------------------------------------------
k = 3
ind_k = indices[indptr[k]:indptr[k+1]]
plt.scatter(vertices[k, 0], vertices[k, 1], color="red", s=100,
            zorder=1)
plt.scatter(vertices[ind_k, 0], vertices[ind_k, 1], color="grey",
            s=100, zorder=1)
# -----------------------------------------------------------------------------
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.show()


###############################################################################
# New inverse distance weighting (IDW) interpolation
###############################################################################

 # Assign triangle vertices to regular grid cells
@njit
def assign_triangles_to_cells(vertices, x_axis, y_axis, dx, dy):
    size_iot = 10
    index_of_tri = np.empty((y_axis.size, x_axis.size, size_iot),
                            dtype=np.int32)
    index_of_tri.fill(-1)
    ind_iot = np.zeros((y_axis.size, x_axis.size), dtype=np.int32)
    dx_half = dx / 2.0
    dx_inv = 1.0 / dx
    dy_half = dy / 2.0
    dy_inv = 1.0 / dy
    for i in range(vertices.shape[0]):    
        # ind_x = int(np.floor((vertices[i, 0] - x_axis[0] + dx_half) * dx_inv))
        ind_x = int((vertices[i, 0] - x_axis[0] + dx_half) * dx_inv)
        if ind_x < 0:
            ind_x = 0
        if ind_x > (x_axis.size - 1):
            ind_x = x_axis.size - 1
        # ind_y = int(np.floor((vertices[i, 1] - y_axis[0] + dy_half) * dy_inv))
        ind_y = int((vertices[i, 1] - y_axis[0] + dy_half) * dy_inv)
        if ind_y < 0:
            ind_y = 0
        if ind_y > (y_axis.size - 1):
            ind_y = y_axis.size - 1
        index_of_tri[ind_y, ind_x, ind_iot[ind_y, ind_x]] = i
        ind_iot[ind_y, ind_x] += 1
        # Increase size of index_of_tri if necessary (e.g. by steps of 2 or 5)
    return index_of_tri, ind_iot

index_of_tri, ind_iot = assign_triangles_to_cells(vertices, x_axis, y_axis, dx, dy)
# %timeit -n 5 -r 5 assign_triangles_to_cells(vertices, x_axis, y_axis, dx, dy)


# Interpolation
num_nn = 4
index = np.empty(num_nn, dtype=np.int32)
dist = np.empty(num_nn)
dist_sqrt_max = 10e9
for ind_y in range(y_axis.size):
    for ind_x in range(x_axis.size):

        index.fill(-1)
        dist.fill(np.nan)
        num_sel = 0

        # ---------------------------------------------------------------------
        # Add points from centre cell
        # ---------------------------------------------------------------------
        for i in range(ind_iot[ind_y, ind_x]):
            ind = index_of_tri[ind_y, ind_x, i]
            dist_sqrt = (x_axis[ind_x] - vertices[ind, 0]) ** 2 \
                + (y_axis[ind_y] - vertices[ind, 0]) ** 2
            # add more stuff here ...
        if num_sel == num_nn:
            continue
        # ---------------------------------------------------------------------
        # Add points from neighbouring cells (in 'frame')
        # ---------------------------------------------------------------------
        level = 1
        while True:

            # Bottom and top
            for i in (-level, +level):  # y-axis
                for j in range(-level, level + 1):  # x-axis
                    ind_y_nb = ind_y + i
                    ind_x_nb = ind_x + j
                    if ((ind_x_nb < 0) or (ind_x_nb > (x_axis.size - 1)) 
                        or (ind_y_nb < 0) or (ind_y_nb > (y_axis.size - 1))):
                        continue
                    print(ind_y_nb, ind_x_nb)
                    for i in range(ind_iot[ind_y_nb, ind_x_nb]):
                        ind = index_of_tri[ind_y_nb, ind_x_nb, i]
                        print(ind, vertices[ind, :])

            # Left and right
            for j in (-level, +level):  # x-axis
                for i in range(-level + 1, level):  # y-axis
                    ind_y_nb = ind_y + i
                    ind_x_nb = ind_x + j
                    if ((ind_x_nb < 0) or (ind_x_nb > (x_axis.size - 1)) 
                        or (ind_y_nb < 0) or (ind_y_nb > (y_axis.size - 1))):
                        continue
                    print(ind_y_nb, ind_x_nb)
                    for i in range(ind_iot[ind_y_nb, ind_x_nb]):
                        ind = index_of_tri[ind_y_nb, ind_x_nb, i]
                        print(ind, vertices[ind, :])


# Test plot
plt.figure()
plt.pcolormesh(x_axis, y_axis, ind_iot, cmap="YlGnBu")
plt.colorbar()
plt.show()
ind_0, ind_1 = np.where(ind_iot == 4)
print(ind_0, ind_1)
print(index_of_tri[1, 4, :])
index = index_of_tri[1, 4, :][index_of_tri[1, 4, :] != -1]
with np.printoptions(precision=2, suppress=True, threshold=5):
    print(vertices[index, :])

###############################################################################
# Interpolation/remapping
###############################################################################

# Bilinear interpolation (regular -> unstructured grid) (Scipy)
f_ip = interpolate.RegularGridInterpolator((x_axis, y_axis), z_reg.transpose(),
                                           method="linear",
                                           bounds_error=False)
z_tri_ip_scipy = f_ip(centroids)

# Bilinear interpolation (regular -> unstructured grid) (Fortran)
z_tri_ip = ip_fortran.bilinear(
    x_axis, y_axis, np.asfortranarray(z_reg),
    np.asfortranarray(centroids))
if np.any(np.any(z_tri_ip == -9999.0)):
    print("Warning: Invalid values in interpolated data")
print(np.abs(z_tri_ip - z_tri_ip_scipy).max())

# IDW interpolation (unstructured -> regular grid) (Scipy)
num_nn = 6  # number of nearest neighbours
kd_tree = KDTree(centroids)
temp = np.meshgrid(x_axis, y_axis)
points_ip = np.vstack((temp[0].ravel(), temp[1].ravel())).T
dist, indices = kd_tree.query(points_ip, k=num_nn, workers=-1)
z_reg_ip_scipy = (((z_tri_ip[indices] / dist)).sum(axis=1)
                / (1.0 / dist).sum(axis=1)).reshape(y_size, x_size)

# IDW interpolation (unstructured -> regular grid) (Fortran)
z_reg_ip = ip_fortran.inverse_distance_weighted(
                np.asfortranarray(centroids.transpose()),
                z_tri_ip, x_axis, y_axis, num_nn)
print(np.abs(z_reg_ip - z_reg_ip_scipy).max())
print(np.abs(z_reg_ip - z_reg_ip_scipy).mean())

