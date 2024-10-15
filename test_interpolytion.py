# Description: Test interpolation between regular and unstructured grid
# (triangle mesh)
#
# Interpolation/remapping:
# regular -> unstructured grid: bilinear interpolation
# unstructured -> regular grid: inverse distance weighting (IDW)
#
# Author: Christian Steger, October 2023

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from scipy import interpolate
from scipy.spatial import KDTree

mpl.style.use("classic") # type: ignore

# Load required Fortran-functions
from interpolation import interpolation as ip_fortran # type: ignore

###############################################################################
# Create example data and grid
###############################################################################

# Regular grid
# ----- small grid/mesh -----
# x_size = 110
# y_size = 91
# num_vertices = 1_300
# ----- large grid/mesh -----
x_size = 770
y_size = 637
num_vertices = 250_000
# ----------------------
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
z_reg[:30, :55] = 15.0

# Triangle mesh
points = np.empty((num_vertices, 2))
points[:, 0] = np.random.uniform(limits["x_min"], limits["x_max"],
                                 num_vertices)
points[:, 1] = np.random.uniform(limits["y_min"], limits["y_max"],
                                 num_vertices)
triangles = Delaunay(points)
vertices = points[triangles.simplices]
centroids = np.mean(vertices, axis=1)
print("Number of cells in triangle mesh: ", centroids.shape[0])

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

# Colormap (absolute)
cmap = plt.get_cmap("Spectral")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(0.0, A)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N) # type: ignore

# Plot absolute values
fontsize = 11.5
plt.figure(figsize=(19.0, 7.0))
gs = gridspec.GridSpec(1, 4, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       wspace=0.05, hspace=0.1,
                       width_ratios=[1.0, 1.0, 1.0, 0.05])
# -----------------------------------------------------------------------------
ax0 = plt.subplot(gs[0])
plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
if centroids.shape[0] < 50_000:
    plt.triplot(points[:, 0], points[:, 1], triangles.simplices, color="black",
                linewidth=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c=cmap(norm(z_tri_ip)), s=50)
plt.axis((limits["x_min"], limits["x_max"], limits["y_min"], limits["y_max"]))
plt.title("Data on regular grid (and interpolated to triangle mesh)",
          fontsize=fontsize)
# -----------------------------------------------------------------------------
ax1 = plt.subplot(gs[1])
plt.tripcolor(points[:, 0], points[:, 1], triangles.simplices, z_tri_ip,
              cmap=cmap, norm=norm)
plt.xticks([])
plt.yticks([])
plt.title("Data interpolated to triangle mesh", fontsize=fontsize)
# -----------------------------------------------------------------------------
ax2 = plt.subplot(gs[2])
plt.pcolormesh(x_axis, y_axis, z_reg_ip, cmap=cmap, norm=norm)
plt.axis((limits["x_min"], limits["x_max"], limits["y_min"], limits["y_max"]))
plt.xticks([])
plt.yticks([])
plt.title("Data re-interpolated to regular grid", fontsize=fontsize)
# -----------------------------------------------------------------------------
ax3 = plt.subplot(gs[3])
cb = mpl.colorbar.ColorbarBase(ax3, cmap=cmap, norm=norm, # type: ignore
                               orientation="vertical")
# -----------------------------------------------------------------------------
plt.show()

# Colormap (difference)
diff = (z_reg_ip - z_reg_ip_scipy)
cmap = plt.get_cmap("RdBu")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=True) \
         .tick_values(np.percentile(diff, 0.1), np.percentile(diff, 0.9))
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")

# Plot difference between Scipy and Fortran implementation
plt.figure()
plt.pcolormesh(x_axis, y_axis, diff, cmap=cmap, norm=norm)
plt.axis((limits["x_min"], limits["x_max"], limits["y_min"], limits["y_max"]))
plt.title("Deviation (Fortran - SciPy) in re-interpolated data",
          fontsize=fontsize)
plt.colorbar()
plt.show()