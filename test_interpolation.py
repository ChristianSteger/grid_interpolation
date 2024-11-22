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

mpl.style.use("classic") # type: ignore

# Load required Fortran-functions
from interpolation import interpolation as ip_fortran # type: ignore

###############################################################################
# Create example triangle mesh and equally spaced regular grid
###############################################################################

# Grid/mesh size
# ----- very small grid/mesh -----
# num_points = 50
# ----- small grid/mesh ---
num_points = 5_000
# ----- large grid/mesh -----
# num_points = 100_000
# ----- very large grid/mesh -----
# num_points = 500_000
# ----------------------

# Create random nodes
points = np.empty((num_points, 2))
np.random.seed(33)
points[:, 0] = np.random.uniform(-0.33, 20.57, num_points)  # x_limits
points[:, 1] = np.random.uniform(-2.47, 9.89, num_points)  # y_limits

# Triangulate nodes
triangles = Delaunay(points)
print("Number of cells in triangle mesh: ", triangles.simplices.shape[0])
indptr_con, indices_con = triangles.vertex_neighbor_vertices

# Generate artificial data on triangle mesh
amplitude = 25.0
def surface(x, y, x_0=points[:, 0].mean(), y_0=points[:, 1].mean(),
            sigma_x=np.std(points[:, 0]), sigma_y=np.std(points[:, 1]),
            amplitude=amplitude):
    surf =  amplitude * np.exp(-((x - x_0) ** 2 / (2 * sigma_x ** 2)) -
                               ((y - y_0) ** 2 / (2 * sigma_y ** 2)))
    # -------------------------------------------------------------------------
    # Modify artificial data further
    # -------------------------------------------------------------------------
    # mask = (points[:, 0] < 6.7) & (points[:, 1] < 1.5)
    # surf[mask] = 18.52
    # -------------------------------------------------------------------------
    return surf
data_pts = surface(points[:, 0], points[:, 1])

# Create equally spaced regular grid
vertices = points[triangles.simplices]  # counterclockwise oriented
temp = np.concatenate((vertices, np.ones(vertices.shape[:2] + (1,))), axis=2)
area_tri = 0.5 * np.linalg.det(temp)
gc_area = area_tri.sum() / triangles.simplices.shape[0]  # mean grid cell area
grid_spac = np.sqrt(gc_area)  # grid spacing (dx = dy)
safety_multi = 1.005
x_ext_pts = (points[:, 0].max() - points[:, 0].min())
y_ext_pts = (points[:, 1].max() - points[:, 1].min())
len_x = int(np.ceil(x_ext_pts * safety_multi / grid_spac))
len_y = int(np.ceil(y_ext_pts * safety_multi / grid_spac))
x_add = ((len_x * grid_spac) - x_ext_pts) / 2.0
x_axis = np.linspace(points[:, 0].min() - x_add, points[:, 0].max() + x_add,
                     len_x + 1)
y_add = ((len_y * grid_spac) - y_ext_pts) / 2.0
y_axis = np.linspace(points[:, 1].min() - y_add, points[:, 1].max() + y_add,
                     len_y + 1)
print("Number of cells in eq. spaced regular grid: ",
      (x_axis.size * y_axis.size))
# -----------------------------------------------------------------------------
# print(np.abs(np.diff(x_axis) - grid_spac).max())
# print(np.abs(np.diff(y_axis) - grid_spac).max())
# print(points[:, 0].min() - x_axis[0])
# print(x_axis[-1] - points[:, 0].max())
# print(points[:, 1].min() - y_axis[0])
# print(y_axis[-1] - points[:, 1].max())
# -----------------------------------------------------------------------------
x_grid = np.linspace(x_axis[0] - grid_spac / 2.0, x_axis[-1] + grid_spac / 2.0,
                     x_axis.size + 1)
y_grid = np.linspace(y_axis[0] - grid_spac / 2.0, y_axis[-1] + grid_spac / 2.0,
                     y_axis.size + 1)

# Colormap (absolute)
cmap = plt.get_cmap("Spectral")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(0.0, amplitude)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N) # type: ignore

# Plot nodes, triangulation and equally spaced regular grid (esrg)
if num_points <= 100_000:
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], triangles.simplices, color="black",
                linewidth=0.5, zorder=3)
    plt.scatter(points[:, 0], points[:, 1], c=data_pts, s=50, zorder=3,
                cmap=cmap, norm=norm)
    plt.colorbar()
    plt.scatter(*np.meshgrid(x_axis, y_axis), c="grey", s=5, zorder=2)
    plt.vlines(x=x_grid, ymin=y_grid[0], ymax=y_grid[-1], colors="black",
               zorder=2)
    plt.hlines(y=y_grid, xmin=x_grid[0], xmax=x_grid[-1], colors="black",
               zorder=2)
    plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
              y_grid[0] - 0.3, y_grid[-1] + 0.3))
    plt.show()

###############################################################################
# Interpolate from unstructured to regular grid
###############################################################################

# IWD, Scipy k-d tree (reference solution)
num_nn = 6  # number of nearest neighbours
kd_tree = KDTree(points)
temp = np.meshgrid(x_axis, y_axis)
points_ip = np.vstack((temp[0].ravel(), temp[1].ravel())).T
dist, indices = kd_tree.query(points_ip, k=num_nn, workers=-1)
data_esrg_scipy = (((data_pts[indices] / dist)).sum(axis=1)
                / (1.0 / dist).sum(axis=1)).reshape(y_axis.size, x_axis.size)

# IWD, Fortran k-d tree
points_ft_trans = np.asfortranarray(points.transpose())
print((" IWD, Fortran k-d tree ").center(79, "-"))
data_esrg_ft = ip_fortran.idw_kdtree(
    points_ft_trans, data_pts, x_axis, y_axis, num_nn)
dev_abs_max = np.abs(data_esrg_ft - data_esrg_scipy).max()
print(f"Maximal absolute deviation: {dev_abs_max:.8f}")
dev_abs_mean = np.abs(data_esrg_ft - data_esrg_scipy).mean()
print(f"Mean absolute deviation: {dev_abs_mean:.8f}")

# IWD, Fortran esrg nearest neighbour + connected points
points_ft = np.asfortranarray(points)
print((" IWD, Fortran esrg nearest neighbour + connected points ")
      .center(79, "-"))
data_esrg_ft = ip_fortran.idw_esrg_connected(
    points_ft, data_pts, x_axis, y_axis, grid_spac,
    (indices_con + 1), (indptr_con + 1))
dev_abs_max = np.abs(data_esrg_ft - data_esrg_scipy).max()
print(f"Maximal absolute deviation: {dev_abs_max:.8f}")
dev_abs_mean = np.abs(data_esrg_ft - data_esrg_scipy).mean()
print(f"Mean absolute deviation: {dev_abs_mean:.8f}")

# IWD, Fortran esrg nearest neighbours
print((" IWD, Fortran esrg nearest neighbours ").center(79, "-"))
data_esrg_ft = ip_fortran.idw_esrg_nearest(
    points_ft, data_pts, x_axis, y_axis, grid_spac, num_nn)
dev_abs_max = np.abs(data_esrg_ft - data_esrg_scipy).max()
print(f"Maximal absolute deviation: {dev_abs_max:.8f}")
dev_abs_mean = np.abs(data_esrg_ft - data_esrg_scipy).mean()
print(f"Mean absolute deviation: {dev_abs_mean:.8f}")

# Barycentric interpolation, Fortran
simplices_ft = np.asfortranarray(triangles.simplices + 1)
# counterclockwise oriented
neighbours_ft = np.asfortranarray(triangles.neighbors + 1) 
print((" Barycentric interpolation, Fortran ").center(79, "-"))
# kth neighbour is opposite to kth vertex
data_bi = ip_fortran.barycentric_interpolation(
    points_ft, data_pts, x_axis, y_axis, simplices_ft, neighbours_ft)

# Plot interpolated field
plt.figure()
# plt.pcolormesh(x_axis, y_axis, data_esrg_ft, cmap=cmap, norm=norm)
plt.pcolormesh(x_axis, y_axis, data_bi, cmap=cmap, norm=norm)
plt.colorbar()
plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
          y_grid[0] - 0.3, y_grid[-1] + 0.3))
plt.show()

###############################################################################
# Interpolate from regular to unstructured grid
###############################################################################

# Scipy (reference solution)
# data_reg = data_esrg_ft
data_reg = data_bi
f_ip = interpolate.RegularGridInterpolator((x_axis, y_axis),
                                           data_reg.transpose(),
                                           method="linear",
                                           bounds_error=False)
data_pts_reip_scipy = f_ip(points)

# Fortran
print((" Bilinear, Fortran ").center(79, "-"))
data_pts_reip_ft = ip_fortran.bilinear(
    x_axis, y_axis, np.asfortranarray(data_reg),
    np.asfortranarray(points))
if np.any(np.any(data_pts_reip_ft == -9999.0)):
    print("Warning: Invalid values in interpolated data")
dev_abs_max = np.abs(data_pts_reip_ft - data_pts_reip_scipy).max()
print(f"Maximal absolute deviation: {dev_abs_max:.8f}")
dev_abs_mean = np.abs(data_pts_reip_ft - data_pts_reip_scipy).mean()
print(f"Mean absolute deviation: {dev_abs_mean:.8f}")

# Compare re-interpolated with original data
print((" Deviation between original and re-interpolated data ")
      .center(79, "-"))
data_dev = data_pts_reip_ft - data_pts
print(f"Maximal absolute deviation: {np.abs(data_dev).max():.8f}")
print(f"Mean absolute deviation: {np.abs(data_dev).mean():.8f}")

# Colormap (difference)
cmap_diff = plt.get_cmap("RdBu")
levels_diff = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=True) \
    .tick_values(np.percentile(data_dev, 0.1),
                 np.percentile(data_dev, 0.9))
norm_diff = mpl.colors.BoundaryNorm( # type: ignore
    levels_diff, ncolors=cmap_diff.N, extend="both")

# Plot original and re-interpolated data and difference
if num_points <= 100_000:
    fontsize = 12.5
    plt.figure(figsize=(16.0, 6.0))
    gs = gridspec.GridSpec(2, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.08, hspace=0.1,
                        height_ratios=[1.0, 0.05])
    # -------------------------------------------------------------------------
    ax0 = plt.subplot(gs[0, 0])
    plt.scatter(points[:, 0], points[:, 1], c=data_pts, s=50, zorder=3,
                cmap=cmap, norm=norm)
    plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
            y_grid[0] - 0.3, y_grid[-1] + 0.3))
    plt.title("Original data", fontsize=fontsize)
    # -------------------------------------------------------------------------
    ax0_c = plt.subplot(gs[1, 0])
    cb = mpl.colorbar.ColorbarBase(ax0_c, cmap=cmap, # type: ignore
                                norm=norm, orientation="horizontal")
    # -------------------------------------------------------------------------
    ax1 = plt.subplot(gs[0, 1])
    plt.scatter(points[:, 0], points[:, 1], c=data_pts_reip_ft, s=50, zorder=3,
                cmap=cmap, norm=norm)
    plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
            y_grid[0] - 0.3, y_grid[-1] + 0.3))
    plt.title("Re-interpolated data", fontsize=fontsize)
    # -------------------------------------------------------------------------
    ax2 = plt.subplot(gs[0, 2])
    plt.scatter(points[:, 0], points[:, 1], c=data_dev, s=50, zorder=3,
                cmap=cmap_diff, norm=norm_diff)
    plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
            y_grid[0] - 0.3, y_grid[-1] + 0.3))
    plt.title("Difference (re-interpolated - original)", fontsize=fontsize)
    # -------------------------------------------------------------------------
    ax2_c = plt.subplot(gs[1, 2])
    cb = mpl.colorbar.ColorbarBase(ax2_c, cmap=cmap_diff, # type: ignore
                                norm=norm_diff, orientation="horizontal")
    # -------------------------------------------------------------------------
    plt.show()
