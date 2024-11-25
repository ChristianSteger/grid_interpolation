# Description: Interpolation based on barycentric coordinates and by walking
#              through the triangle mesh (-> development of Python
#              implementation)
#
# Author: Christian Steger, November 2024

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.spatial import Delaunay

mpl.style.use("classic") # type: ignore

###############################################################################
# Functions and classes
###############################################################################

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class LineSegment:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2

# -----------------------------------------------------------------------------
# Check location of point relative to line
# -----------------------------------------------------------------------------

# https://stackoverflow.com/questions/1560492/how-to-tell-whether-a-point-is
# -to-the-right-or-left-side-of-a-line#comment1419666_1560510

def triangle_point_outside(edge, point_in):
    """Checks, based on a single edge of a triangle, if a point is outside of
    the triangle (the vertices must be oriented counterclockwise)"""

    # margin = 0.0
    margin = -1e-10  # 'safety margin' (0.0 or slightly smaller)
    cross_prod = (edge.pt2.x - edge.pt1.x) * (point_in.y - edge.pt1.y) \
        - (edge.pt2.y - edge.pt1.y) * (point_in.x - edge.pt1.x)
    return (cross_prod < margin)

# Test function
vertices = (Point(3.0, 2.0), Point(9.0, 3.0), Point(5.0, 7.0))
# P = Point(5.0, 5.0)
# P = Point(7.0, 7.0)
P = Point(3.0, 2.0 - 1e-15)
# P = Point(4.079, 4.7)

# Check point location relative to triangle
point_outside = False
for i in range(3):
    edge = LineSegment(vertices[i], vertices[(i + 1) % 3])
    if triangle_point_outside(edge, P):
        point_outside = True

# Plot
plt.figure()
for i in range(3):
    plt.plot([vertices[i].x, vertices[(i + 1) % 3].x],
             [vertices[i].y, vertices[(i + 1) % 3].y],
             color="grey", lw=1.5, zorder=1)
    plt.text(vertices[i].x + 0.2, vertices[i].y + 0.2,
             f"V{i + 1}", fontsize=12)
plt.scatter([i.x for i in vertices], [i.y for i in vertices],
            s=150, color="black", zorder=2)
plt.scatter(P.x, P.y, s=75, color="red", zorder=3)
plt.title(f"Point outside of triangle: {point_outside}")
plt.show()

# -----------------------------------------------------------------------------
# Triangle walk function
# -----------------------------------------------------------------------------

# https://inria.hal.science/inria-00072509/document
# -> visibility walk (always terminates for a Delaunay triangulation)

def triangle_walk(points, simplices, neighbours, neighbour_none,
                  point_target, ind_tri, plot, print_stat):
    iters = 0
    point_inside_ch = True
    # Plot start --------------------------------------------------------------
    if plot:
        cols = np.array(["green", "blue", "red"])
        plt.figure(figsize=(10 * 1.2, 7 * 1.2))
        ax = plt.axes()
        plt.triplot(points[:, 0], points[:, 1], simplices, color="black",
                    linewidth=0.5)
        centroids = points[simplices].mean(axis=1)
        plt.scatter(centroids[:, 0], centroids[:, 1], color="black",
                    marker="+", s=50)
        plt.scatter(points[simplices[ind_tri, :], 0],
                    points[simplices[ind_tri, :], 1],
                    color=cols, s=50)
        ind_neigh = neighbours[ind_tri]
        mask = (ind_neigh != neighbour_none)
        plt.scatter(centroids[ind_neigh[mask], 0],
                    centroids[ind_neigh[mask], 1],
                    color=cols[mask], marker="o", s=100, alpha=0.5)
        plt.scatter(point_target.x, point_target.y, color="black", s=100,
                    marker="*")
        plt.scatter(*centroids[ind_tri, :], color="black", s=50)
    # Plot end ----------------------------------------------------------------
    while True:

        iters += 1
        # Plot start ----------------------------------------------------------
        if plot:
            poly = list(zip(points[simplices[ind_tri, :], 0],
                            points[simplices[ind_tri, :], 1]))
            poly = plt.Polygon(poly, facecolor="orange", # type: ignore
                            edgecolor="none", alpha=0.5, zorder=-2)
            ax.add_patch(poly)
        # Plot end ------------------------------------------------------------

        # Find intersection with triangle edge
        point_outside = False
        for i in range(3):
            ind_1 = i
            ind_2 = (i + 1) % 3
            line_edge = LineSegment(
                Point(*points[simplices[ind_tri, ind_1], :]), 
                Point(*points[simplices[ind_tri, ind_2], :])
                )
            if triangle_point_outside(line_edge, point_target):
                point_outside = True
                break
        if point_outside == False:
            # Plot start ------------------------------------------------------
            if plot:
                poly = list(zip(points[simplices[ind_tri, :], 0],
                                points[simplices[ind_tri, :], 1]))
                poly = plt.Polygon(poly, facecolor="red", # type: ignore
                                edgecolor="none", alpha=0.5, zorder=-2)
                ax.add_patch(poly)
            # Plot end --------------------------------------------------------
            if print_stat:
                print("Exit: Triangle containing point found")
            break
        i -= 1
        if i < 0:
            i = 2
        ind_tri_pre = ind_tri
        ind_tri = neighbours[ind_tri, i]

        # Points outside of convex hull
        if ind_tri == neighbour_none:
            if print_stat:
                print("Point is outside of convex hull")
            ind_tri = ind_tri_pre  # set triangle index to last valid
            point_inside_ch = False
            break

    # Plot start --------------------------------------------------------------
    if plot:
        plt.title(f"Number of iterations" + f": {iters}", loc="left")
        plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
                y_grid[0] - 0.3, y_grid[-1] + 0.3))
        plt.show()
    # Plot end ----------------------------------------------------------------
    return ind_tri, point_inside_ch, iters

# -----------------------------------------------------------------------------
# Weights for barycentric interpolation
# -----------------------------------------------------------------------------

# https://codeplea.com/triangular-interpolation

def barycentric_interpolation(vertex_1, vertex_2, vertex_3, point):

    denom = (vertex_2.y - vertex_3.y) * (vertex_1.x - vertex_3.x) \
        + (vertex_3.x - vertex_2.x) * (vertex_1.y - vertex_3.y)
    weight_vt1 = ((vertex_2.y - vertex_3.y) * (point.x - vertex_3.x)
                  + (vertex_3.x - vertex_2.x) * (point.y - vertex_3.y)) / denom
    weight_vt2 = ((vertex_3.y - vertex_1.y) * (point.x - vertex_3.x)
                  + (vertex_1.x - vertex_3.x) * (point.y - vertex_3.y)) / denom
    weight_vt3 = 1.0 - weight_vt1 - weight_vt2

    return weight_vt1, weight_vt2, weight_vt3

# -----------------------------------------------------------------------------
# Fill cells in rectangular grid with nearest neighbour
# -----------------------------------------------------------------------------

def fill_nearest_neighbour(mask_outside, x_axis, y_axis, data):

    for ind_y in range(y_axis.size):
        for ind_x in range(x_axis.size):
            if mask_outside[ind_y, ind_x]:
                print("-" * 60)
                print(f"Cell requiring filling ({ind_y}, {ind_x})")
                level = 0
                nn_found = False
                dist_sq_sel = np.inf
                while not nn_found:
                    level += 1
                    radius_sq_min = (level - 0.5) ** 2
                    radius_sq_max = (level + 0.5) ** 2
                    print("Considered radius range: " 
                          + "[%.1f" % np.sqrt(radius_sq_min) + ", " 
                          + "%.1f" % np.sqrt(radius_sq_max) + "]")
                    for i in range(np.maximum(ind_y - level, 0),
                                   np.minimum(ind_y + level + 1, y_axis.size)):
                        for j in range(np.maximum(ind_x - level, 0),
                                       np.minimum(ind_x + level + 1,
                                                  x_axis.size)):
                            if not mask_outside[i, j]:
                                dist_sq = (i - ind_y) ** 2 + (j - ind_x) ** 2
                                if ((dist_sq >= radius_sq_min) 
                                    and (dist_sq < radius_sq_max) 
                                    and (dist_sq < dist_sq_sel)):
                                    print("(" + str(i) + ", " + str(j) 
                                          + ") dist.: %.2f" % np.sqrt(dist_sq))
                                    dist_sq_sel = dist_sq
                                    data[ind_y, ind_x] = data[i, j]
                                    nn_found = True
                print(f"Final selected distance: {np.sqrt(dist_sq_sel):.2f}")

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
# ----- small grid/mesh -----
num_points = 5_000
# ----- large grid/mesh -----
# num_points = 100_000
# ----------------------

# Create random nodes
points = np.empty((num_points, 2))
np.random.seed(33)
points[:, 0] = np.random.uniform(-0.33, 20.57, num_points)  # x_limits
points[:, 1] = np.random.uniform(-2.47, 9.89, num_points)  # y_limits

# Triangulate nodes
triangles = Delaunay(points)
print("Number of cells in triangle mesh: ", triangles.simplices.shape[0])

# Generate artificial data on triangle mesh
# data_pts = gaussian_mountain(x=points[:, 0], y=points[:, 1],
#                              x_0=points[:, 0].mean(),
#                              y_0=points[:, 1].mean(),
#                              sigma_x=np.std(points[:, 0]),
#                              sigma_y=np.std(points[:, 1]))
data_pts = sin_mountains(x=points[:, 0], y=points[:, 1])

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
# Test barycentric interpolation (and filling of rectangular grid cells
# outside of convex hull with nearest neighbours)
###############################################################################

# Arrays describing triangle mesh configuration
simplices = triangles.simplices  # counterclockwise oriented
neighbours = triangles.neighbors # kth neighbour is opposite to kth vertex
neighbour_none = -1  # value denoting no neighbour

# Settings for triangle walk algorithm
ind_tri_start = 34  # starting triangle
# point_target = Point(3.81, 7.48)
# point_target = Point(18.4, 0.09)
# point_target = Point(20.61, -2.18)  # outside of convex hull (ch)
point_target = Point(np.random.uniform(x_grid[0], x_grid[-1]),
                     np.random.uniform(y_grid[0], y_grid[-1]))
# point_target = Point(*points[np.random.randint(0, num_points), :])
# point_target = Point(20.540, 7.053)  # outside of ch, move along hull
# point_target = Point(0.697768, 1.287293)   # outside of ch, move along hull
# point_target = Point(17.91493, -2.67667)   # outside of ch, move along hull

# Call function
ind_tri_out, point_inside_ch, iters \
    = triangle_walk(points, simplices, neighbours, neighbour_none,
                    point_target, ind_tri_start, plot=False, print_stat=True)

# Overview plot
if point_inside_ch:
    plt.figure(figsize=(10 * 1.2, 7 * 1.2))
    ax = plt.axes()
    plt.triplot(points[:, 0], points[:, 1], simplices, color="black",
                linewidth=0.5)
    cols = ("red", "darkgreen")
    for ind_i, i in enumerate((ind_tri_start, ind_tri_out)):
        poly = list(zip(points[simplices[i, :], 0],
                        points[simplices[i, :], 1]))
        poly = plt.Polygon(poly, facecolor=cols[ind_i], # type: ignore
                        edgecolor="none",
                        alpha=0.4, zorder=-2)
        ax.add_patch(poly)
    centroids = points[simplices].mean(axis=1)
    plt.scatter(centroids[:, 0], centroids[:, 1], color="black",
                marker="+", s=50)
    plt.scatter(point_target.x, point_target.y, s=100, marker="o",
                color="green", zorder=5)
    # -------------------------------------------------------------------------
    # Barycentric interpolation weights
    # -------------------------------------------------------------------------
    ind = simplices[ind_tri_out, :]
    vertex_1 = Point(*points[ind[0], :])
    vertex_2 = Point(*points[ind[1], :])
    vertex_3 = Point(*points[ind[2], :])
    weight_vt1, weight_vt2, weight_vt3 \
        = barycentric_interpolation(vertex_1, vertex_2, vertex_3, point_target)
    plt.scatter(vertex_1.x, vertex_1.y, s=weight_vt1 * 200.0, color="black")
    plt.scatter(vertex_2.x, vertex_2.y, s=weight_vt2 * 200.0, color="black")
    plt.scatter(vertex_3.x, vertex_3.y, s=weight_vt3 * 200.0, color="black")
    # -------------------------------------------------------------------------
    plt.show()

# Barycentric interpolation for rectangular grid
ind_tri_start = 0
iters_all = 0
data_ip = np.empty((y_axis.size, x_axis.size))
data_ip.fill(np.nan)
mask_outside = np.zeros((y_axis.size, x_axis.size), dtype=bool)
for ind_y in range(y_axis.size):
    # -------------------------------
    # ind_x_start = 0
    # ind_x_end = x_axis.size
    # ind_x_step = +1
    # -------------------------------
    if (ind_y % 2 == 0):
        ind_x_start = 0
        ind_x_end = x_axis.size
        ind_x_step = +1
    else:
        ind_x_start = x_axis.size - 1
        ind_x_end = -1
        ind_x_step = -1
    # -------------------------------
    for ind_x in range(ind_x_start, ind_x_end, ind_x_step):
        point_target = Point(x_axis[ind_x], y_axis[ind_y])
        ind_tri_out, point_inside_ch, iters \
            = triangle_walk(points, simplices, neighbours, neighbour_none,
                            point_target, ind_tri_start,
                            plot=False, print_stat=False)
        ind_tri_start = ind_tri_out  # start from previous triangle
        # ---------------------------------------------------------------------
        if point_inside_ch:
            ind = simplices[ind_tri_out, :]
            vertex_1 = Point(*points[ind[0], :])
            vertex_2 = Point(*points[ind[1], :])
            vertex_3 = Point(*points[ind[2], :])
            weight_vt1, weight_vt2, weight_vt3 \
                = barycentric_interpolation(vertex_1, vertex_2,
                                            vertex_3, point_target)
            data_ip[ind_y, ind_x] = data_pts[ind[0]] * weight_vt1 \
                                + data_pts[ind[1]] * weight_vt2 \
                                + data_pts[ind[2]] * weight_vt3
        else:
            mask_outside[ind_y, ind_x] = True
        # ---------------------------------------------------------------------
        iters_all += iters

    print(ind_y)
print(f"Number of iterations: {iters_all}")
ipc = iters_all / (x_axis.size * y_axis.size)
print(f"Iterations per interpolated cell: {ipc:.2f}")

# Fill missing cells in rectangular grid with nearest neighbour
data_ip_cp = data_ip.copy()
fill_nearest_neighbour(mask_outside, x_axis, y_axis, data_ip)

# Test plot
plt.figure()
# plt.pcolormesh(x_grid, y_grid, data_ip_cp, cmap=cmap, norm=norm)
plt.pcolormesh(x_grid, y_grid, data_ip, cmap=cmap, norm=norm)
plt.scatter(*np.meshgrid(x_axis, y_axis), c="grey", s=10)
plt.triplot(points[:, 0], points[:, 1], triangles.simplices, color="black",
            linewidth=0.5, zorder=3)
plt.scatter(points[:, 0], points[:, 1], c=data_pts, s=50, zorder=3,
            cmap=cmap, norm=norm)
plt.colorbar(cmap=cmap, norm=norm)
plt.show()
