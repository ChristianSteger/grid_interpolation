# Description: Interpolation based on barycentric coordinates and by walking
#              through the triangle mesh (-> development of Python 
#              implementation)
#
# Author: Christian Steger, October 2024

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.spatial import Delaunay

mpl.style.use("classic") # type: ignore

###############################################################################
# Two-dimensional line/point algorithms and auxiliary functions
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
# Line intersection check
# -----------------------------------------------------------------------------

# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
# -> Collinearity does not occur in the 'triangle walk' because the centroid
#    is always used as a strating point. Therefore, the faster below algorithm
#    can be used:
# https://stackoverflow.com/questions/3838329/
# how-can-i-check-if-two-segments-intersect

def ccw(A, B, C):
    """Check if points A, B, C are counterclockwise oriented."""
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def linesegments_intersect(line_1, line_2):
    """Check if line segments intersect (~returns False if line segments 
       only touch)"""
    # rccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    return ccw(line_1.pt1, line_2.pt1, line_2.pt2) \
        != ccw(line_1.pt2, line_2.pt1, line_2.pt2) \
            and ccw(line_1.pt1, line_1.pt2, line_2.pt1) \
                != ccw(line_1.pt1, line_1.pt2, line_2.pt2)

# Test function
# line_1 = LineSegment(Point(1.0, 1.0), Point(2.0, 4.0))
line_1 = LineSegment(Point(1.0, 1.0), Point(2.0, 4.5))
line_2 = LineSegment(Point(1.0, 5.0), Point(5.0, 1.0))
inters = linesegments_intersect(line_1, line_2)

# Plot
plt.figure()
for line in (line_1, line_2):
    plt.plot([line.pt1.x, line.pt2.x], [line.pt1.y, line.pt2.y],
             color="black", lw=1.5)
    plt.scatter([line.pt1.x, line.pt2.x], [line.pt1.y, line.pt2.y],
                s=20, color="black")
plt.axis((0, 6, 0, 6))
plt.title(f"Line segments intersect: {inters}")
plt.show()

# -----------------------------------------------------------------------------
# Find base point (-> shorest distance bewteen line and point)
# -----------------------------------------------------------------------------

def base_point(line, point):
    """Find the base point, which marks the shortest distance bewteen 
       a line and a point)"""
    dist_x = line.pt2.x - line.pt1.x
    dist_y = line.pt2.y - line.pt1.y
    u = (dist_x * (point.x - line.pt1.x) + dist_y * (point.y - line.pt1.y)) \
        / (dist_x ** 2 + dist_y ** 2)
    point_base = Point(line.pt1.x + dist_x * u, line.pt1.y + dist_y * u)
    return u, point_base

# Test function
line = LineSegment(Point(3.0, 6.0), Point(6.0, 3.0))
# line = LineSegment(Point(2.0, 5.0), Point(6.0, 5.0))  # horizontal
# line = LineSegment(Point(5.0, 2.0), Point(5.0, 6.0))  # vertical
point = Point(4.5, 3.0)
u, point_base = base_point(line, point)

# Plot
plt.figure(figsize=(8, 8))
plt.plot([line.pt1.x, line.pt2.x], [line.pt1.y, line.pt2.y],
         color="black", lw=1.5)
plt.scatter([line.pt1.x, line.pt2.x, point.x],
            [line.pt1.y, line.pt2.y, point.y],
            s=20, color="black")
plt.text(line.pt1.x + 0.2, line.pt1.y + 0.2, "Start", fontsize=12)
plt.text(line.pt2.x + 0.2, line.pt2.y + 0.2, "End", fontsize=12)
plt.scatter([point_base.x], [point_base.y], s=20, color="red")
plt.axis([0.0, 8.0, 0.0, 8.0])
plt.title(f"u = {u:.2f}")
plt.show()

# -----------------------------------------------------------------------------
# Auxiliary functions
# -----------------------------------------------------------------------------

def get_rem(array, elem_1, elem_2):
    """Get remaining element from array/list with 3 elements"""
    for i in range(3):
        if (array[i] != elem_1) and (array[i] != elem_2):
            return array[i]

def distance_sq(A, B):
    """Compute squared distance between point A and B"""
    return (A.x - B.x) ** 2 + (A.y - B.y) ** 2

# -----------------------------------------------------------------------------
# Weights for barycentric interpolation)
# -----------------------------------------------------------------------------

# https://codeplea.com/triangular-interpolation

def barycentric_interpolation(vt1, vt2, vt3, point):

    denom = (vt2.y - vt3.y) * (vt1.x - vt3.x) \
        + (vt3.x - vt2.x) * (vt1.y - vt3.y)
    weight_vt1 = ((vt2.y - vt3.y) * (point.x - vt3.x) 
                    + (vt3.x - vt2.x) * (point.y - vt3.y)) / denom
    weight_vt2 = ((vt3.y - vt1.y) * (point.x - vt3.x) 
                    + (vt1.x - vt3.x) * (point.y - vt3.y)) / denom
    weight_vt3 = 1.0 - weight_vt1 - weight_vt2

    return weight_vt1, weight_vt2, weight_vt3

###############################################################################
# Create example triangle mesh and equally spaced regular grid
###############################################################################

# Grid/mesh size
# ----- very small grid/mesh -----
# num_points = 50
# ----- small grid/mesh -----
# num_points = 5_000
# ----- large grid/mesh -----
num_points = 100_000
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
# Walk through mesh to find triangle
###############################################################################

# -----------------------------------------------------------------------------
# Function to move to ajdacent edge in convex hull
# -----------------------------------------------------------------------------

def get_adjacent_edge(simplices, neighbours, neighbour_none,
                      ind_tri, ind_rot, ind_opp):
    ind_vtx_rot = simplices[ind_tri, ind_rot]  # constant
    while True:
        ind_vtx_opp = simplices[ind_tri, ind_opp]
        ind_vtx_rem = get_rem(simplices[ind_tri, :],
                              ind_vtx_rot, ind_vtx_opp)
        if neighbours[ind_tri, ind_opp] == neighbour_none:
            break
        else:
            ind_tri = neighbours[ind_tri, ind_opp]
            ind_opp = np.where(ind_vtx_rem == simplices[ind_tri, :])[0][0]
    ind_rot = np.where(ind_vtx_rot == simplices[ind_tri, :])[0][0]
    ind_rem = np.where(ind_vtx_rem == simplices[ind_tri, :])[0][0]
    return ind_tri, ind_rot, ind_opp, ind_rem

# -----------------------------------------------------------------------------
# Triangle walk function
# -----------------------------------------------------------------------------

def triangle_walk(points, simplices, neighbours, neighbour_none,
                  ind_tri_start, point_target, plot, print_stat):
    ind_loop = [0, 1, 2, 0]
    tri_visited = np.zeros(simplices.shape[0], dtype=bool)
    ind_tri = ind_tri_start
    inters_found = True
    iters = 0
    iters_ch = 0  # iteration along convex hull
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
        plt.scatter(points[simplices[ind_tri_start, :], 0],
                    points[simplices[ind_tri_start, :], 1],
                    color=cols, s=50)
        ind_neigh = neighbours[ind_tri_start]
        mask = (ind_neigh != neighbour_none)
        plt.scatter(centroids[ind_neigh[mask], 0],
                    centroids[ind_neigh[mask], 1],
                    color=cols[mask], marker="o", s=100, alpha=0.5)
        plt.scatter(point_target.x, point_target.y, color="black", s=50)
        plt.scatter(*centroids[ind_tri_start, :], color="black", s=50)
    # Plot end ----------------------------------------------------------------
    while True:

        if not tri_visited[ind_tri]:
            tri_visited[ind_tri] = True
        else:
            # Plot start ------------------------------------------------------
            if plot:
                poly = list(zip(points[simplices[ind_tri, :], 0],
                                points[simplices[ind_tri, :], 1]))
                poly = plt.Polygon(poly, facecolor="red",
                                edgecolor="none", alpha=0.5, zorder=-2)
                ax.add_patch(poly)
            # Plot end --------------------------------------------------------
            if print_stat:
                print("Exit: Triangle already visited")
            ind_tri_out = ind_tri
            point_out = point_target
            break
        iters += 1
        centroid = Point(*points[simplices[ind_tri, :], :].mean(axis=0))
        line_walk = LineSegment(centroid, point_target)
        # Plot start ----------------------------------------------------------
        if plot:
            poly = list(zip(points[simplices[ind_tri, :], 0],
                            points[simplices[ind_tri, :], 1]))
            poly = plt.Polygon(poly, facecolor="orange",
                            edgecolor="none", alpha=0.5, zorder=-2)
            ax.add_patch(poly)
            plt.plot([line_walk.pt1.x, line_walk.pt2.x],
                    [line_walk.pt1.y, line_walk.pt2.y],
                    color="orange", linestyle="-", lw=0.8, zorder=-1)
        # Plot end ------------------------------------------------------------

        # Find intersection with triangle edges
        inters_found = False
        for i in range(3):
            ind_1 = ind_loop[i]
            ind_2 = ind_loop[i + 1]
            line_edge = LineSegment(
                Point(*points[simplices[ind_tri, ind_1], :]),
                Point(*points[simplices[ind_tri, ind_2], :])
                )
            if linesegments_intersect(line_walk, line_edge):
                inters_found = True
                break
        if inters_found == False:
            # Plot start ------------------------------------------------------
            if plot:
                poly = list(zip(points[simplices[ind_tri, :], 0],
                                points[simplices[ind_tri, :], 1]))
                poly = plt.Polygon(poly, facecolor="red",
                                edgecolor="none", alpha=0.5, zorder=-2)
                ax.add_patch(poly)
            # Plot end --------------------------------------------------------
            if print_stat:
                print("Exit: Triangle containing point found")
            ind_tri_out = ind_tri
            point_out = point_target
            break
        i -= 1
        if i < 0:
            i = 2
        ind_tri_pre = ind_tri
        ind_tri = neighbours[ind_tri, i]

        # Handle points outside of convex hull
        if ind_tri == neighbour_none:
            if print_stat:
                print("Point is outside of convex hull")
            ind_tri = ind_tri_pre  # set triangle index to last valid
            u, point_base = base_point(line_edge, point_target)
            if (u >= 0) and (u <= 1):
                # -------------------------------------------------------------
                # Point is perpendicular to 'direct outer' edge of triangle
                # -------------------------------------------------------------
                # Plot start --------------------------------------------------
                if plot:
                    plt.plot([point_target.x, point_base.x], 
                            [point_target.y, point_base.y],
                            color="black",linestyle=":", lw=0.8, zorder=-1)
                    plt.scatter(point_base.x, point_base.y, color="black",
                                marker="*", s=50)
                    poly = list(zip(points[simplices[ind_tri, :], 0],
                                    points[simplices[ind_tri, :], 1]))
                    poly = plt.Polygon(poly, facecolor="red",
                                    edgecolor="none", alpha=0.5, zorder=-2)
                    ax.add_patch(poly)
                # Plot end ----------------------------------------------------
                if print_stat:
                    print("Exit: Point is perpendicular to 'direct outer' "
                        + "edge of triangle")
                ind_tri_out = ind_tri
                point_out = point_base
                break

            else:
                # -------------------------------------------------------------
                # Point is not perpendicular to 'direct outer' edge of triangle
                # -------------------------------------------------------------
                if print_stat:
                    print("Point is not perpendicular to 'direct outer' edge"
                        + " of triangle")
                
                # Define 'rotation' and 'opposite' vertices
                dist_sq_1 = distance_sq(point_target, line_edge.pt1)
                dist_sq_2 = distance_sq(point_target, line_edge.pt2)
                if (dist_sq_1 < dist_sq_2):
                    ind_rot = ind_1
                    ind_opp = ind_2
                else:
                    ind_rot = ind_2
                    ind_opp = ind_1

                # Move along line segments of convex hull
                while True:

                    iters_ch += 1

                    # Move to triangle that represents adjacent outer edge
                    ind_tri, ind_rot, ind_opp, ind_rem \
                        = get_adjacent_edge(simplices, neighbours, 
                                            neighbour_none,
                                            ind_tri, ind_rot, ind_opp)

                    # Plot start ----------------------------------------------
                    if plot:
                        poly = list(zip(
                            points[simplices[ind_tri, :], 0],
                            points[simplices[ind_tri, :], 1]
                            ))
                        poly = plt.Polygon(poly, facecolor="blue",
                                        edgecolor="none", alpha=0.5, zorder=-2)
                        ax.add_patch(poly)
                        plt.scatter(*points[simplices[ind_tri, ind_rot], :],
                                    s=30, marker="o", color="blue")
                        plt.scatter(*points[simplices[ind_tri, ind_rem], :],
                                    s=30, marker="^", color="blue")
                    # Plot end ------------------------------------------------

                    line_edge = LineSegment(
                        Point(*points[simplices[ind_tri, ind_rot], :]),
                        Point(*points[simplices[ind_tri, ind_rem], :])
                        )

                    # Check if point is perpendicular to edge
                    u, point_base = base_point(line_edge, point_target)
                    if (u >= 0) and (u <= 1):
                        # Plot start ------------------------------------------
                        if plot:
                            plt.plot([point_target.x, point_base.x], 
                                     [point_target.y, point_base.y],
                                     color="black",linestyle=":", lw=0.8,
                                     zorder=-1)
                            plt.scatter(point_base.x, point_base.y,
                                        color="black", marker="*", s=50)
                            poly = list(zip(
                                points[simplices[ind_tri, :], 0],
                                points[simplices[ind_tri, :], 1]
                                ))
                            poly = plt.Polygon(poly, facecolor="red",
                                            edgecolor="none", alpha=0.5,
                                            zorder=-2)
                            ax.add_patch(poly)
                        # Plot end --------------------------------------------
                        if print_stat:
                            print("Exit: Point is perpendicular to edge")
                        ind_tri_out = ind_tri
                        point_out = point_base
                        break

                    # Check if point should be assigned to 'rotation vertices'
                    dist_sq_1 = distance_sq(point_target, line_edge.pt1)
                    dist_sq_2 = distance_sq(point_target, line_edge.pt2)
                    if dist_sq_1 < dist_sq_2:
                        # Plot start ------------------------------------------
                        if plot:
                            plt.scatter(
                                *points[simplices[ind_tri, ind_rot], :],
                                s=150, facecolors="none", edgecolors="black"
                                )
                            poly = list(zip(
                                points[simplices[ind_tri, :], 0],
                                points[simplices[ind_tri, :], 1]
                                ))
                            poly = plt.Polygon(poly, facecolor="red",
                                            edgecolor="none", alpha=0.5,
                                            zorder=-2)
                            ax.add_patch(poly)
                        # Plot end --------------------------------------------
                        if print_stat:
                            print("Exit: Point is not perpendicular to edge "
                                + "-> use nearest vertices")
                        ind_tri_out = ind_tri
                        point_out \
                            = Point(*points[simplices[ind_tri, ind_rot], :])
                        break

                    # Move to next line of convex hull during next iteration
                    ind_opp = ind_rot
                    ind_rot = ind_rem

                break
                # point assigned in inner 'while loop' -> break out of outer

    # Plot start --------------------------------------------------------------
    if plot:
        plt.title(f"Number of iterations" + f": {iters}", loc="left")
        plt.title(f"Number of iterations along convex hull" + f": {iters_ch}",
                  loc="right")
        plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
                y_grid[0] - 0.3, y_grid[-1] + 0.3))
        plt.show()
    # Plot end ----------------------------------------------------------------
    return ind_tri_out, point_out, iters
# -----------------------------------------------------------------------------

# Arrays describing triangle mesh configuration
simplices = triangles.simplices  # counterclockwise oriented
neighbours = triangles.neighbors # kth neighbour is opposite to kth vertex
neighbour_none = -1  # value denoting no neighbour

# Settings for triangle walk algorithm
ind_tri_start = 34  # starting triangle
point_target = Point(4.99, 0.55)
# point_target = Point(3.81, 7.48)
# point_target = Point(18.4, 0.09)
# point_target = Point(20.61, -2.18)  # outside of convex hull (ch)
# point_target = Point(np.random.uniform(x_grid[0], x_grid[-1]),
#                      np.random.uniform(y_grid[0], y_grid[-1]))
# point_target = Point(*points[np.random.randint(0, num_points), :])
# point_target = Point(20.540, 7.053)  # outside of ch, move along hull
# point_target = Point(0.697768, 1.287293)   # outside of ch, move along hull
# point_target = Point(17.91493, -2.67667)   # outside of ch, move along hull

# Call function
ind_tri_out, point_out, iters \
    = triangle_walk(points, simplices, neighbours, neighbour_none,
                    ind_tri_start, point_target, plot=False, print_stat=True)

# Overview plot
plt.figure(figsize=(10 * 1.2, 7 * 1.2))
ax = plt.axes()
plt.triplot(points[:, 0], points[:, 1], simplices, color="black",
            linewidth=0.5)
cols = ("red", "darkgreen")
for ind_i, i in enumerate((ind_tri_start, ind_tri_out)):
    poly = list(zip(points[simplices[i, :], 0],
                    points[simplices[i, :], 1]))
    poly = plt.Polygon(poly, facecolor=cols[ind_i], edgecolor="none",
                       alpha=0.4, zorder=-2)
    ax.add_patch(poly)
centroids = points[simplices].mean(axis=1)
plt.scatter(centroids[:, 0], centroids[:, 1], color="black",
            marker="+", s=50)
plt.scatter(point_target.x, point_target.y, s=100, marker="o", color="green",
            zorder=5)
plt.scatter(point_out.x, point_out.y, s=100, marker="*", color="blue",
            zorder=5)
# -----------------------------------------------------------------------------
# Barycentric interpolation weights
# -----------------------------------------------------------------------------
ind = simplices[ind_tri_out, :]
vt1 = Point(*points[ind[0], :])
vt2 = Point(*points[ind[1], :])
vt3 = Point(*points[ind[2], :])
weight_vt1, weight_vt2, weight_vt3 \
    = barycentric_interpolation(vt1, vt2, vt3, point_out)
plt.scatter(vt1.x, vt1.y, s=weight_vt1 * 200.0, color="black")
plt.scatter(vt2.x, vt2.y, s=weight_vt2 * 200.0, color="black")
plt.scatter(vt3.x, vt3.y, s=weight_vt3 * 200.0, color="black")
# -----------------------------------------------------------------------------
plt.show()

# Barycentric interpolation for regctangular grid
ind_tri_start = 0
iters_all = 0
data_ip = np.empty((y_axis.size, x_axis.size))
for ind_y in range(y_axis.size):
    # -------------------------------
    # indices_x = range(x_axis.size)
    # -------------------------------
    if (ind_y % 2 == 0):
        indices_x = range(x_axis.size)
    else:
        indices_x = range(x_axis.size - 1, -1, -1)
    # -------------------------------
    for ind_x in indices_x:
        point_target = Point(x_axis[ind_x], y_axis[ind_y])
        ind_tri_out, point_out, iters \
            = triangle_walk(points, simplices, neighbours, neighbour_none,
                            ind_tri_start, point_target,
                            plot=False, print_stat=False)
        ind_tri_start = ind_tri_out  # start from previous triangle
        # ---------------------------------------------------------------------
        ind = simplices[ind_tri_out, :]
        vt1 = Point(*points[ind[0], :])
        vt2 = Point(*points[ind[1], :])
        vt3 = Point(*points[ind[2], :])
        weight_vt1, weight_vt2, weight_vt3 \
            = barycentric_interpolation(vt1, vt2, vt3, point_out)
        data_ip[ind_y, ind_x] = data_pts[ind[0]] * weight_vt1 \
                              + data_pts[ind[1]] * weight_vt2 \
                              + data_pts[ind[2]] * weight_vt3
        # ---------------------------------------------------------------------
        iters_all += iters

    print(ind_y)
print(f"Number of iterations: {iters_all}")
ipc = iters_all / (x_axis.size * y_axis.size)
print(f"Iterations per interpolated cell: {ipc:.2f}")

# Test plot
plt.figure()
plt.pcolormesh(x_grid, y_grid, data_ip, cmap=cmap, norm=norm)
plt.colorbar()
plt.show()