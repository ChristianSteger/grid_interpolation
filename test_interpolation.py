# Description: Test interpolation between regular and unstructured grid
# (triangle mesh)
#
# Interpolation/remapping:
# - regular -> unstructured grid: bilinear interpolation
# - unstructured -> regular grid: inverse distance weighting (IDW)
# - triangle mesh -> regular grid: barycentric interpolation
#
# Author: Christian Steger, November 2024

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay
from scipy import interpolate
from scipy.spatial import KDTree
from time import perf_counter
from shapely.geometry import Polygon
import shapely.vectorized

mpl.style.use("classic") # type: ignore

# Load required Fortran-functions
from interpolation import interpolation as ip_fortran # type: ignore

###############################################################################
# Functions for artificial 2-dimensional surfaces
###############################################################################

def gaussian_mountain(x, y, x_0, y_0, sigma_x, sigma_y, amplitude=25.0):
    surf =  amplitude * np.exp(-((x - x_0) ** 2 / (2 * sigma_x ** 2)) -
                               ((y - y_0) ** 2 / (2 * sigma_y ** 2)))
    return surf

def sin_mountains(x, y, omega=0.6, amplitude=25.0):
    surf = amplitude * np.sin(omega * x + y) + amplitude
    return surf

###############################################################################
# Create example triangle mesh and equally spaced regular grid
###############################################################################

# Grid/mesh size
# ----- very small grid/mesh -----
# num_points = 50
# ----- small grid/mesh ---
# num_points = 5_000
# ----- large grid/mesh -----
# num_points = 100_000
# ----- very large grid/mesh -----
num_points = 500_000
# ----------------------

# Create random nodes
points = np.empty((num_points, 2))
np.random.seed(33)
points[:, 0] = np.random.uniform(-0.33, 20.57, num_points)  # x_limits
points[:, 1] = np.random.uniform(-2.47, 9.89, num_points)  # y_limits

# Triangulate nodes
triangles = Delaunay(points)
print("Number of cells in triangle mesh: ", triangles.simplices.shape[0])
simplices = triangles.simplices
neighbors = triangles.neighbors

# Generate artificial data on triangle mesh
# data_pts = gaussian_mountain(x=points[:, 0], y=points[:, 1],
#                              x_0=points[:, 0].mean(),
#                              y_0=points[:, 1].mean(),
#                              sigma_x=np.std(points[:, 0]),
#                              sigma_y=np.std(points[:, 1]))
data_pts = sin_mountains(x=points[:, 0], y=points[:, 1])

# Create equally spaced regular grid
vertices = points[simplices]  # counterclockwise oriented
temp = np.concatenate((vertices, np.ones(vertices.shape[:2] + (1,))), axis=2)
area_tri = 0.5 * np.linalg.det(temp)
gc_area = area_tri.sum() / simplices.shape[0]  # mean grid cell area
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
         .tick_values(0.0, data_pts.max())
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N) # type: ignore

# Plot nodes, triangulation and equally spaced regular grid (esrg)
if num_points <= 100_000:
    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], simplices, color="black",
                linewidth=0.5, zorder=3)
    plt.scatter(points[:, 0], points[:, 1], c=data_pts, s=50, zorder=3,
                cmap=cmap, norm=norm)
    plt.colorbar()
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
print(" Interpolate from unstructured to regular grid ".center(79, "#"))

# -----------------------------------------------------------------------------
# Scipy reference solutions
# -----------------------------------------------------------------------------
print("Reference solutions (Scipy):")

# Inverse distance weighting (IDW)
num_nn = 6  # number of nearest neighbours
time_1 = perf_counter()
kd_tree = KDTree(points)
points_ip = np.vstack([i.ravel() for i in np.meshgrid(x_axis, y_axis)]).T
dist, indices = kd_tree.query(points_ip, k=num_nn, workers=-1)
data_reg_iwd_ref = (((data_pts[indices] / dist)).sum(axis=1)
                    / (1.0 / dist).sum(axis=1)).reshape(y_axis.size,
                                                        x_axis.size)
time_2 = perf_counter()
print(f"IDW interpolation: {(time_2 - time_1):.2f} s")

# Barycentric interpolation
time_1 = perf_counter()
data_reg_bi_ref = interpolate.griddata(points, data_pts, points_ip,
                                       method="linear")
data_reg_bi_ref = data_reg_bi_ref.reshape(y_axis.size, x_axis.size)
time_2 = perf_counter()
print(f"Barycentric interpolation: {(time_2 - time_1):.2f} s")

# -----------------------------------------------------------------------------
# Fortran solutions
# -----------------------------------------------------------------------------
print("Fortran solutions:")

def deviation_stats(data, data_ref, mask=None, mask_name=None,
                    percentiles=[99.99, 99.0]):
    if mask_name is None:
        txt = "Absolute deviation statistics:"
    else:
        txt = "Absolute deviation statistics (" + mask_name + "):"
    print(txt)
    if mask is None:
        dev_abs = np.abs(data - data_ref)
    else:
        dev_abs = np.abs(data[mask] - data_ref[mask])
    print(f"- Maximum: {np.nanmax(dev_abs):.5f}")
    for q in percentiles:
        dev_abs_per = np.nanpercentile(dev_abs, q)
        print(f"- {q}th percentile: {dev_abs_per:.5f}")
    print(f"- Mean: {np.nanmean(dev_abs):.5f}")

# Inverse distance weighting (IDW), k-d tree
print((" IDW interpolation:, k-d tree ").center(60, "-"))
points_ft_trans = np.asfortranarray(points.transpose())
data_reg_iwd_kdtree_ft = ip_fortran.idw_kdtree(
    points_ft_trans, data_pts, x_axis, y_axis, num_nn)
deviation_stats(data_reg_iwd_kdtree_ft, data_reg_iwd_ref)

# Inverse distance weighting (IDW), esrg nearest neighbours
print((" IDW interpolation:, esrg nearest neighbours ").center(60, "-"))
points_ft = np.asfortranarray(points)
data_reg_iwd_esrg_ft = ip_fortran.idw_esrg_nearest(
    points_ft, data_pts, x_axis, y_axis, grid_spac, num_nn)
deviation_stats(data_reg_iwd_esrg_ft, data_reg_iwd_ref)

# Barycentric interpolation
print((" Barycentric interpolation ").center(60, "-"))
simplices_ft = np.asfortranarray(simplices + 1)
# counterclockwise oriented
neighbours_ft = np.asfortranarray(neighbors + 1)
# kth neighbour is opposite to kth vertex
data_reg_bi_ft = ip_fortran.barycentric_interpolation(
    points_ft, data_pts, x_axis, y_axis, simplices_ft, neighbours_ft)
deviation_stats(data_reg_bi_ft, data_reg_bi_ref)

# Plot interpolated field
plt.figure()
# plt.pcolormesh(x_axis, y_axis, data_reg_iwd_esrg_ft, cmap=cmap, norm=norm)
plt.pcolormesh(x_axis, y_axis, data_reg_bi_ft, cmap=cmap, norm=norm)
plt.colorbar()
plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
          y_grid[0] - 0.3, y_grid[-1] + 0.3))
plt.show()

###############################################################################
# Interpolate from regular to unstructured grid
###############################################################################
print(" Interpolate from regular to unstructured grid ".center(79, "#"))

# Order line segments of convex hull
convex_hull = triangles.convex_hull
convex_hull_ord = convex_hull.copy()
convex_hull[0, :] = -1
for i in range(1, convex_hull_ord.shape[0]):
    j, k = np.where(convex_hull_ord[i - 1, 1] == convex_hull[:, :])
    if k[0] == 0:
        convex_hull_ord[i, :] = convex_hull[j[0], :]
    else:
        convex_hull_ord[i, :] = convex_hull[j[0], :][::-1]
    convex_hull[j[0], :] = -1
del convex_hull

# Mask with 'inner' points
poly_ch = Polygon(zip(points[convex_hull_ord[:, 0], 0],
                      points[convex_hull_ord[:, 0], 1]))
resolution = convex_hull_ord.shape[0] * 5
poly_ch_small = poly_ch.buffer(-0.25, resolution=resolution)
mask_pts_in = shapely.vectorized.contains(poly_ch_small, points[:, 0],
                                          points[:, 1])

# Scipy reference solutions
data_reg = {"iwd": data_reg_iwd_esrg_ft, "bi": data_reg_bi_ft}
data_pts_rec = {}
for i in data_reg.keys():
    f_ip = interpolate.RegularGridInterpolator((x_axis, y_axis),
                                            data_reg[i].transpose(),
                                            method="linear",
                                            bounds_error=False)
    data_pts_rec[i] = f_ip(points)

# Fortran solution
print((" Fortran bilinear interpolation ").center(60, "-"))
data_pts_rec_ft = ip_fortran.bilinear(
    x_axis, y_axis, np.asfortranarray(data_reg_iwd_esrg_ft),
    np.asfortranarray(points))
if np.any(np.any(data_pts_rec_ft == -9999.0)):
    print("Warning: Invalid values in interpolated data")
deviation_stats(data_pts_rec_ft, data_pts_rec["iwd"])

# Compare re-interpolated with original data
methods = {"iwd": "inverse distance weighting",
           "bi": "barycentric interpolation"}
for i in data_reg.keys():
    print((" Re-interpolation: " + methods[i] + " ").center(60, "-"))
    deviation_stats(data_pts_rec[i], data_pts)
    deviation_stats(data_pts_rec[i], data_pts, mask_pts_in,
                    mask_name="inner domain")
print("-" * 60)

# Colormap (difference)
data_dev = (data_pts_rec["bi"] - data_pts)
cmap_diff = plt.get_cmap("RdBu")
levels_diff = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=True) \
    .tick_values(np.percentile(data_dev, 0.25),
                 np.percentile(data_dev, 0.75))
norm_diff = mpl.colors.BoundaryNorm( # type: ignore
    levels_diff, ncolors=cmap_diff.N, extend="both")

# Plot original and re-interpolated data and difference
if num_points <= 100_000:
    fontsize = 12.5
    plt.figure(figsize=(16.0, 6.0))
    gs = gridspec.GridSpec(2, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                           wspace=0.08, hspace=0.1, height_ratios=[1.0, 0.05])
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
    plt.scatter(points[:, 0], points[:, 1], c=data_pts_rec["bi"], s=50,
                zorder=3, cmap=cmap, norm=norm)
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
    plt.plot(*poly_ch.exterior.coords.xy, color="black", lw=1.5, zorder=4)
    plt.plot(*poly_ch_small.exterior.coords.xy, color="black", lw=1.5,
             zorder=4)
    # -------------------------------------------------------------------------
    ax2_c = plt.subplot(gs[1, 2])
    cb = mpl.colorbar.ColorbarBase(ax2_c, cmap=cmap_diff, # type: ignore
                                   norm=norm_diff, orientation="horizontal")
    # -------------------------------------------------------------------------
    plt.show()
