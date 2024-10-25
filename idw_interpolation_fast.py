# Description: Fast inverse distance weighted (IDW) interpolation methods to
#              to regular grid with equal spacing in both dimensions.
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
# x_size = 10
# y_size = 8
# num_points = 50
# ----- small grid/mesh -----
x_size = 110
y_size = 91
num_points = 5_000
# ----- large grid/mesh -----
# x_size = 770
# y_size = 637
# num_points = 250_000
# ----------------------

# Regular grid
grid_spac = 10
# dx != dy so that below interpolation methods work correctly!!!
x_axis = np.arange(-50.0, -50.0 + grid_spac * x_size, grid_spac)
y_axis = np.arange(-30.0, -30.0 + grid_spac * y_size, grid_spac)
print("Number of cells in regular grid: ", (x_axis.size * y_axis.size))

# Example data on regular grid (2-dimensional Gaussian distribution)
x_0 = x_axis.mean()
y_0 = y_axis.mean()
sigma_x = (x_axis.mean() - x_axis[0]) / 2.0
sigma_y = (y_axis.mean() - y_axis[0]) / 2.0
A = 25.0
z_reg = A * np.exp(-((x_axis[np.newaxis, :] - x_0) ** 2 / (2 * sigma_x ** 2)) -
                   ((y_axis[:, np.newaxis] - y_0) ** 2 / (2 * sigma_y ** 2)))
# z_reg[:20, :30] = 15.0

# Grid lines
x_grid = np.linspace(x_axis[0] - grid_spac / 2.0, x_axis[-1] + grid_spac / 2.0,
                     x_axis.size + 1)
y_grid = np.linspace(y_axis[0] - grid_spac / 2.0, y_axis[-1] + grid_spac / 2.0,
                     y_axis.size + 1)

# Triangle mesh
# -----------------------------------------------------------------------------
points = np.empty((num_points, 2))
np.random.seed(33)
points[:, 0] = np.random.uniform(x_axis[0], x_axis[-1], num_points)
points[:, 1] = np.random.uniform(y_axis[0], y_axis[-1], num_points)
# -----------------------------------------------------------------------------
# x = np.linspace(x_axis[0], x_axis[-1], 80)
# y = np.linspace(y_axis[0], y_axis[-1], 60)
# X, Y = np.meshgrid(x, y)
# points = np.vstack([X.ravel(), Y.ravel()]).T
# -----------------------------------------------------------------------------
triangles = Delaunay(points)
print("Number of cells in triangle mesh: ", triangles.simplices.shape[0])
# vertices_per_triangle = vertices[triangles.simplices]
# neighbors = triangles.neighbors
# neighbour triangles (-1: boundary -> no triangle)
indptr, indices = triangles.vertex_neighbor_vertices

###############################################################################
# Bilinear interpolation to points and overview plot
###############################################################################

# Interpolate to points
f_ip = interpolate.RegularGridInterpolator((x_axis, y_axis), z_reg.transpose(),
                                           method="linear",
                                           bounds_error=False)
z_reg_ip = f_ip(points)

# Colormap (absolute)
cmap = plt.get_cmap("Spectral")
levels = MaxNLocator(nbins=20, steps=[1, 2, 5, 10], symmetric=False) \
         .tick_values(0.0, A)
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N) # type: ignore

# Plot
plt.figure()
plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
plt.colorbar()
plt.vlines(x=x_grid, ymin=y_grid[0], ymax=y_grid[-1], colors="black")
plt.hlines(y=y_grid, xmin=x_grid[0], xmax=x_grid[-1], colors="black")
# -----------------------------------------------------------------------------
plt.triplot(points[:, 0], points[:, 1], triangles.simplices, color="black",
                linewidth=0.5, zorder=2)
plt.scatter(points[:, 0], points[:, 1], color="black", s=25, zorder=2)
# -----------------------------------------------------------------------------
# plt.scatter(points[:, 0], points[:, 1], c=cmap(norm(z_reg_ip)), s=150,
#             zorder=3)
# -----------------------------------------------------------------------------
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.show()

###############################################################################
 # Assign points to regular grid cells
###############################################################################

 # Assign points to regular grid cells
@njit
def assign_points_to_cells(points, x_axis, y_axis, grid_spac):
    size_iot = 10
    index_of_pts = np.empty((y_axis.size, x_axis.size, size_iot),
                            dtype=np.int32)
    index_of_pts.fill(-1)
    num_iot = np.zeros((y_axis.size, x_axis.size), dtype=np.int32)
    for i in range(points.shape[0]):
        ind_x = int((points[i, 0] - x_axis[0] + grid_spac / 2.0) 
                    * 1.0 / grid_spac)
        if ind_x < 0:
            ind_x = 0
        if ind_x > (x_axis.size - 1):
            ind_x = x_axis.size - 1
        ind_y = int((points[i, 1] - y_axis[0] + grid_spac / 2.0) 
                    * 1.0 / grid_spac)
        if ind_y < 0:
            ind_y = 0
        if ind_y > (y_axis.size - 1):
            ind_y = y_axis.size - 1
        index_of_pts[ind_y, ind_x, num_iot[ind_y, ind_x]] = i
        num_iot[ind_y, ind_x] += 1
        # Increase size of index_of_pts if necessary (e.g. by steps of 5)...
    return index_of_pts, num_iot

index_of_pts, num_iot = assign_points_to_cells(points, x_axis, y_axis,
                                               grid_spac)
# %timeit -n 5 -r 5 assign_points_to_cells(points, x_axis, y_axis, grid_spac)

###############################################################################
# IDW interpolation from 'n' nearest neighbours
###############################################################################

def idw_neighbours_n(points, data_in, x_axis, y_axis, grid_spac, num_nn):

    index_of_pts, num_iot = assign_points_to_cells(points, x_axis, y_axis,
                                               grid_spac)
    
    # K-d tree as reference
    kd_tree = KDTree(points)

    index_nn = np.empty(num_nn, dtype=np.int32)
    dist_sq_nn = np.empty(num_nn)
    data_ip = np.empty((y_axis.size, x_axis.size))
    for ind_y in range(y_axis.size):
        for ind_x in range(x_axis.size):

            # ind_x = 45
            # ind_y = 56

            index_nn.fill(-1)
            dist_sq_nn.fill(np.inf)
            centre = (x_axis[ind_x], y_axis[ind_y])

            # -----------------------------------------------------------------
            # Centre cell
            # -----------------------------------------------------------------

            level = 0
            radius_sq = (grid_spac * (level + 0.5)) ** 2
        
            # Assign points to list
            points_cons = []
            for i in range(num_iot[ind_y, ind_x]):
                ind = index_of_pts[ind_y, ind_x, i]
                dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                        + (centre[1] - points[ind, 1]) ** 2
                points_cons.append((ind, dist_sq))

            # Potentially add new points
            points_cons_next = []
            for ind, dist_sq in points_cons:
                if (dist_sq > radius_sq):
                    points_cons_next.append((ind, dist_sq))
                elif (dist_sq < dist_sq_nn[-1]):
                    ind_is = (dist_sq_nn < dist_sq).sum() # insertion index
                    dist_sq_nn[ind_is + 1:] = dist_sq_nn[ind_is:-1]
                    dist_sq_nn[ind_is] = dist_sq
                    index_nn[ind_is + 1:] = index_nn[ind_is:-1]
                    index_nn[ind_is] = ind

            num_nn_sel = (index_nn != -1).sum()

            # -----------------------------------------------------------------
            # Neighbouring cells of 'frame'
            # -----------------------------------------------------------------
            while num_nn_sel != num_nn:

                level += 1
                radius_sq = (grid_spac * (level + 0.5)) ** 2

                # Assign points from bottom and top cells to list
                points_cons = points_cons_next.copy()
                for i in (-level, +level):  # y-axis
                    for j in range(-level, level + 1):  # x-axis
                        ind_y_nb = ind_y + i
                        ind_x_nb = ind_x + j
                        if ((ind_x_nb < 0)
                            or (ind_x_nb > (x_axis.size - 1)) 
                            or (ind_y_nb < 0) 
                            or (ind_y_nb > (y_axis.size - 1))):
                            continue
                        for k in range(num_iot[ind_y_nb, ind_x_nb]):
                            ind = index_of_pts[ind_y_nb, ind_x_nb, k]
                            dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                                    + (centre[1] - points[ind, 1]) ** 2
                            points_cons.append((ind, dist_sq))

                # Assign points from left and right cells to list
                for j in (-level, +level):  # x-axis
                    for i in range(-level + 1, level):  # y-axis
                        ind_y_nb = ind_y + i
                        ind_x_nb = ind_x + j
                        if ((ind_x_nb < 0)
                            or (ind_x_nb > (x_axis.size - 1)) 
                            or (ind_y_nb < 0) 
                            or (ind_y_nb > (y_axis.size - 1))):
                            continue
                        for k in range(num_iot[ind_y_nb, ind_x_nb]):
                            ind = index_of_pts[ind_y_nb, ind_x_nb, k]
                            dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                                    + (centre[1] - points[ind, 1]) ** 2
                            points_cons.append((ind, dist_sq))

                # Potentially add new points
                points_cons_next = []
                for ind, dist_sq in points_cons:
                    if (dist_sq > radius_sq):
                        points_cons_next.append((ind, dist_sq))
                    elif(dist_sq < dist_sq_nn[-1]):
                        ind_is = (dist_sq_nn < dist_sq).sum() # insertion index
                        dist_sq_nn[ind_is + 1:] = dist_sq_nn[ind_is:-1]
                        dist_sq_nn[ind_is] = dist_sq
                        index_nn[ind_is + 1:] = index_nn[ind_is:-1]
                        index_nn[ind_is] = ind               

                num_nn_sel = (index_nn != -1).sum()

            # Compare with k-d tree
            dist_kd, indices_kd = kd_tree.query(centre, k=num_nn)
            if (not np.all(index_nn == indices_kd)):
                raise ValueError("Error: (order of) point indices differ")
            if (np.abs(dist_sq_nn - (dist_kd ** 2)).max() > 1e-10):
                raise ValueError("Error: NN-distances differ significantly")

            # Interpolate data
            dist_nn = np.sqrt(dist_sq_nn)
            if dist_nn[0] == 0.0:  # avoid division by zero
                data_ip[ind_y, ind_x] = data_in[index_nn[0]]
            else:
                data_ip[ind_y, ind_x] = (((data_in[index_nn] / dist_nn).sum())
                                         / (1.0 / dist_nn).sum())

            # # Plot
            # plt.figure(figsize=(8.8, 7))
            # ax = plt.axes()
            # plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
            # plt.vlines(x=x_grid, ymin=y_grid[0], ymax=y_grid[-1],
            #            colors="black")
            # plt.hlines(y=y_grid, xmin=x_grid[0], xmax=x_grid[-1],
            #            colors="black")
            # # ---------------------------------------------------------------
            # plt.triplot(points[:, 0], points[:, 1], triangles.simplices,
            #             color="black", linewidth=0.5, zorder=2)
            # plt.scatter(points[:, 0], points[:, 1], color="black", s=25,
            #             zorder=2)
            # # ---------------------------------------------------------------
            # ind_in = np.array([ind for ind, dist_sq in points_cons])
            # if ind_in.size > 0:
            #     plt.scatter(points[ind_in, 0], points[ind_in, 1],
            #                 color="green", s=100, zorder=1)
            # circle = plt.Circle(centre, np.sqrt(radius_sq), # type: ignore
            #                     facecolor="none", edgecolor="green", lw=1.5)
            # ax.add_patch(circle)
            # # ---------------------------------------------------------------
            # ind_out = np.array([ind for ind, dist_sq in points_cons_next])
            # if ind_out.size > 0:
            #     plt.scatter(points[ind_out, 0], points[ind_out, 1],
            #                 color="blue", s=100, zorder=1)
            # # ---------------------------------------------------------------
            # plt.scatter(x_axis[ind_x], y_axis[ind_y], color="red", s=200,
            #             marker="*", zorder=1)
            # index_nn_tmp = index_nn[index_nn != -1]
            # plt.scatter(points[index_nn_tmp, 0], points[index_nn_tmp, 1],
            #             color="red", s=100, zorder=1)
            # radius = np.sqrt(dist_sq_nn[(num_nn_sel - 1)])
            # circle = plt.Circle(centre, radius, # type: ignore
            #                     facecolor="none", edgecolor="red", lw=1.5)
            # ax.add_patch(circle)
            # # ---------------------------------------------------------------
            # plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
            # plt.show()

    return data_ip

num_nn = 6
data_in = z_reg_ip
data_ip_nn = idw_neighbours_n(points, data_in, x_axis, y_axis, grid_spac, num_nn)
# %timeit -n 1 -r 1 idw_neighbours_n(points, data_in, x_axis, y_axis, grid_spac, num_nn)

# Compare initial with re-interpolated data
fontsize = 13.0
plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       wspace=0.1, hspace=0.1,
                       width_ratios=[1.0, 1.0, 0.05])
# -----------------------------------------------------------------------------
ax0 = plt.subplot(gs[0])
plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.title("Initial data on regular grid", fontsize=fontsize)
# -----------------------------------------------------------------------------
ax1 = plt.subplot(gs[1])
plt.pcolormesh(x_axis, y_axis, data_ip_nn, cmap=cmap, norm=norm)
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.title("re-interpolated data (num_nn = " + str(num_nn) + ")",
          fontsize=fontsize)
plt.text(0.15, 0.07, f"MAE: {np.abs(data_ip_nn - z_reg).mean():.3f}",
         horizontalalignment="center", verticalalignment="center",
         transform = ax1.transAxes, fontweight="bold")
# -----------------------------------------------------------------------------
ax2 = plt.subplot(gs[2])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, # type: ignore
                               orientation="vertical")
# -----------------------------------------------------------------------------
plt.show()

###############################################################################
# IDW interpolation from nearest neighbour and points connected via 
# triangulation
###############################################################################

def idw_neighbours_con(points, data_in, x_axis, y_axis, grid_spac):

    index_of_pts, num_iot = assign_points_to_cells(points, x_axis, y_axis,
                                               grid_spac)
    
    # K-d tree as reference
    kd_tree = KDTree(points)

    data_ip = np.empty((y_axis.size, x_axis.size))
    for ind_y in range(y_axis.size):
        for ind_x in range(x_axis.size):

            # ind_x = 45
            # ind_y = 56

            index_nn = -1
            dist_sq_nn = np.inf
            centre = (x_axis[ind_x], y_axis[ind_y])
            num_nn_sel = 0

            # -----------------------------------------------------------------
            # Centre cell
            # -----------------------------------------------------------------

            level = 0
            radius_sq = (grid_spac * (level + 0.5)) ** 2
            
            # Assign points to list
            points_cons = []
            for i in range(num_iot[ind_y, ind_x]):
                ind = index_of_pts[ind_y, ind_x, i]
                dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                        + (centre[1] - points[ind, 1]) ** 2
                points_cons.append((ind, dist_sq))

            # Potentially add new points
            points_cons_next = []
            for ind, dist_sq in points_cons:
                if (dist_sq > radius_sq):
                    points_cons_next.append((ind, dist_sq))
                elif (dist_sq < dist_sq_nn):
                    dist_sq_nn = dist_sq
                    index_nn = ind
                    num_nn_sel = 1

            # -----------------------------------------------------------------
            # Neighbouring cells of 'frame'
            # -----------------------------------------------------------------
            while num_nn_sel == 0:

                level += 1
                radius_sq = (grid_spac * (level + 0.5)) ** 2

                # Assign points from bottom and top cells to list
                points_cons = points_cons_next.copy()
                for i in (-level, +level):  # y-axis
                    for j in range(-level, level + 1):  # x-axis
                        ind_y_nb = ind_y + i
                        ind_x_nb = ind_x + j
                        if ((ind_x_nb < 0)
                            or (ind_x_nb > (x_axis.size - 1)) 
                            or (ind_y_nb < 0)
                            or (ind_y_nb > (y_axis.size - 1))):
                            continue
                        for k in range(num_iot[ind_y_nb, ind_x_nb]):
                            ind = index_of_pts[ind_y_nb, ind_x_nb, k]
                            dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                                    + (centre[1] - points[ind, 1]) ** 2
                            points_cons.append((ind, dist_sq))

                # Assign points from left and right cells to list
                for j in (-level, +level):  # x-axis
                    for i in range(-level + 1, level):  # y-axis
                        ind_y_nb = ind_y + i
                        ind_x_nb = ind_x + j
                        if ((ind_x_nb < 0)
                            or (ind_x_nb > (x_axis.size - 1)) 
                            or (ind_y_nb < 0)
                            or (ind_y_nb > (y_axis.size - 1))):
                            continue
                        for k in range(num_iot[ind_y_nb, ind_x_nb]):
                            ind = index_of_pts[ind_y_nb, ind_x_nb, k]
                            dist_sq = (centre[0] - points[ind, 0]) ** 2 \
                                    + (centre[1] - points[ind, 1]) ** 2
                            points_cons.append((ind, dist_sq))

                # Potentially add new points
                points_cons_next = []
                for ind, dist_sq in points_cons:
                    if (dist_sq > radius_sq):
                        points_cons_next.append((ind, dist_sq))
                    elif (dist_sq < dist_sq_nn):
                        dist_sq_nn = dist_sq
                        index_nn = ind
                        num_nn_sel = 1      

            # Compare with k-d tree
            dist_kd, indices_kd = kd_tree.query(centre, k=1)
            if (not (index_nn == indices_kd)):
                raise ValueError("Error: (order of) point indices differ")
            if (np.abs(dist_sq_nn - (dist_kd ** 2)) > 1e-10):
                raise ValueError("Error: NN-distances differ significantly")

            # Get indices / squared distances of connected points
            ind_con = indices[indptr[index_nn]:indptr[index_nn + 1]]
            dist_sq_con = (centre[0] - points[ind_con, 0]) ** 2 \
                        + (centre[1] - points[ind_con, 1]) ** 2
            index_nn = np.append(index_nn, ind_con)
            dist_sq_nn = np.append(dist_sq_nn, dist_sq_con)

            # Interpolate data
            dist_nn = np.sqrt(dist_sq_nn)
            if dist_nn[0] == 0.0:  # avoid division by zero
                data_ip[ind_y, ind_x] = data_in[index_nn[0]]
            else:
                data_ip[ind_y, ind_x] = (((data_in[index_nn] / dist_nn).sum())
                                            / (1.0 / dist_nn).sum())

            # # Plot
            # plt.figure(figsize=(8.8, 7))
            # ax = plt.axes()
            # plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
            # plt.vlines(x=x_grid, ymin=y_grid[0], ymax=y_grid[-1],
            #            colors="black")
            # plt.hlines(y=y_grid, xmin=x_grid[0], xmax=x_grid[-1],
            #            colors="black")
            # # ---------------------------------------------------------------
            # plt.triplot(points[:, 0], points[:, 1], triangles.simplices,
            #             color="black", linewidth=0.5, zorder=2)
            # plt.scatter(points[:, 0], points[:, 1], color="black", s=25,
            #             zorder=2)
            # # ---------------------------------------------------------------
            # ind_in = np.array([ind for ind, dist_sq in points_cons])
            # if ind_in.size > 0:
            #     plt.scatter(points[ind_in, 0], points[ind_in, 1],
            #                 color="green", s=100, zorder=1)
            # circle = plt.Circle(centre, np.sqrt(radius_sq), # type: ignore
            #                     facecolor="none", edgecolor="green", lw=1.5)
            # ax.add_patch(circle)
            # # ---------------------------------------------------------------
            # ind_out = np.array([ind for ind, dist_sq in points_cons_next])
            # if ind_out.size > 0:
            #     plt.scatter(points[ind_out, 0], points[ind_out, 1],
            #                 color="blue", s=100, zorder=1)
            # # ---------------------------------------------------------------
            # plt.scatter(x_axis[ind_x], y_axis[ind_y], color="red", s=200,
            #             marker="*", zorder=1)
            # index_nn_tmp = index_nn[index_nn != -1]
            # plt.scatter(points[index_nn_tmp, 0], points[index_nn_tmp, 1],
            #             color="red", s=100, zorder=1)
            # plt.scatter(points[index_nn_tmp[0], 0],
            #             points[index_nn_tmp[0], 1],
            #             color="red", s=250, zorder=1)
            # circle = plt.Circle(centre, # type: ignore
            #                     np.sqrt(dist_sq_nn[(num_nn_sel - 1)]),
            #                     facecolor="none", edgecolor="red", lw=1.5)
            # ax.add_patch(circle)
            # # ---------------------------------------------------------------
            # plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
            # plt.show()

    return data_ip

data_in = z_reg_ip
data_ip_nc = idw_neighbours_con(points, data_in, x_axis, y_axis, grid_spac)
# %timeit -n 1 -r 1 idw_neighbours_con(points, data_in, x_axis, y_axis, grid_spac)

# Compare initial with re-interpolated data
fontsize = 13.0
plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 3, left=0.1, bottom=0.1, right=0.9, top=0.9,
                       wspace=0.1, hspace=0.1,
                       width_ratios=[1.0, 1.0, 0.05])
# -----------------------------------------------------------------------------
ax0 = plt.subplot(gs[0])
plt.pcolormesh(x_axis, y_axis, z_reg, cmap=cmap, norm=norm)
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.title("Initial data on regular grid", fontsize=fontsize)
# -----------------------------------------------------------------------------
ax1 = plt.subplot(gs[1])
plt.pcolormesh(x_axis, y_axis, data_ip_nc, cmap=cmap, norm=norm)
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.title("re-interpolated data (nearest neighbour + connected points)",
          fontsize=fontsize)
plt.text(0.15, 0.07, f"MAE: {np.abs(data_ip_nc - z_reg).mean():.3f}",
         horizontalalignment="center", verticalalignment="center",
         transform = ax1.transAxes, fontweight="bold")
# -----------------------------------------------------------------------------
ax2 = plt.subplot(gs[2])
cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, # type: ignore
                               orientation="vertical")
# -----------------------------------------------------------------------------
plt.show()

###############################################################################
# Check difference between interpolation methods
###############################################################################

plt.figure(figsize=(8.8, 7))
plt.pcolormesh(x_axis, y_axis, data_ip_nc - data_ip_nn, cmap="RdBu")
plt.axis((x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]))
plt.colorbar()
plt.show()