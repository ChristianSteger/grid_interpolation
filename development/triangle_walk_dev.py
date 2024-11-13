# Description: Find triangle for regular grid cell centre by waling through
#              mesh (-> development of Python implementation)
#
# Author: Christian Steger, October 2024

# Load modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
# import matplotlib.gridspec as gridspec
from scipy.spatial import Delaunay

mpl.style.use("classic") # type: ignore

###############################################################################
# Create example triangle mesh and equally spaced regular grid
###############################################################################

# Grid/mesh size
# ----- very small grid/mesh -----
num_points = 50
# ----- small grid/mesh -----
# num_points = 5_000
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
# Algorithm to check if two lines intersect
# -----------------------------------------------------------------------------
# https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

class Point: 
    def __init__(self, x, y): 
        self.x = x 
        self.y = y 

# -> Below solution does not check for collinearity
# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect
def ccw(A, B, C):
    """Check if points A, B, C are counterclockwise oriented."""
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def intersect(A, B, C, D):
    """Check if line segments AB and CD intersect. Returns False if line
    segments only touch"""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

A = Point(4, 6)
B = Point(2, 3)
C = Point(7, 3)
print(ccw(A, B, C))
print(ccw(B, A, C))

# Test data for line segments 'AB' and 'CD'
A = Point(1, 1)
# A = Point(3, 3 + 1e-10)
# B = Point(4, 2 + 1e-15)
# B = Point(2 + 1e-15, 4)
B = Point(2, 4)
C = Point(1, 5)
D = Point(5, 1)

# Plot
plt.figure()
plt.plot([A.x, B.x], [A.y, B.y], color="black")
plt.plot([C.x, D.x], [C.y, D.y], color="black")
x = [A.x, B.x, C.x, D.x]
y = [A.y, B.y, C.y, D.y]
plt.scatter(x, y, color="black", s=20)
labels = ["A", "B", "C", "D"]
for i in range(4):
    plt.text(x[i] + 0.1, y[i] + 0.1, labels[i], fontsize=12)
plt.axis((0, 6, 0, 6))
plt.title(f"Line segments AB and CD intersect: {intersect(A, B, C, D)}")
plt.show()

# %timeit intersect(A, B, C, D)

# -----------------------------------------------------------------------------


simplices = triangles.simplices  # counterclockwise oriented
vertices = points[simplices]  # counterclockwise oriented
neighbours = triangles.neighbors # kth neighbour is opposite to kth vertex

centroids = vertices.mean(axis=1)
cols = np.array(["green", "blue", "red"])

# Settings for triangle walk algorithm
ind_tri = 34  # starting triangle (in centre)
# ind_tri = 45  # starting triangle (at edge)
# point_target = (4.99, 0.55)
# point_target = (3.81, 7.48)
# point_target = (18.4, 0.09)
# point_target = (20.61, -2.18)  # outside of convex hull
# point_target = (np.random.uniform(x_grid[0], x_grid[-1]),
#                 np.random.uniform(y_grid[0], y_grid[-1]))
# point_target = points[np.random.randint(0, num_points), :]
point_target = (20.540, 7.053)  # special case: not yet implemented !!!

# Plot
plt.figure(figsize=(10 * 1.5, 7 * 1.5))
ax = plt.axes()
plt.triplot(points[:, 0], points[:, 1], simplices, color="black",
            linewidth=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], color="black", marker="+", s=50)
plt.scatter(vertices[ind_tri, :, 0], vertices[ind_tri, :, 1],
            color=cols, s=50)
ind_neigh = neighbours[ind_tri]
mask = ind_neigh != -1
plt.scatter(centroids[ind_neigh[mask], 0], centroids[ind_neigh[mask], 1],
            color=cols[mask], marker="o", s=100, alpha=0.5)
plt.scatter(*point_target, color="black", s=50)
plt.scatter(*centroids[ind_tri, :], color="black", s=50)
# -----------------------------------------------------------------------------
# Triangle walk
# -----------------------------------------------------------------------------
ind_loop = [0, 1, 2, 0]
pt_end = Point(*point_target)
ind_cur = ind_tri
inters_found = True
pt_outside_hull = False
tri_visited = np.zeros(simplices.shape[0], dtype=bool)
count = 0
while True:

    if not tri_visited[ind_cur]:
        tri_visited[ind_cur] = True
    else:
        print("Break: Triangle already visited")
        break
    count += 1
    pt_start = Point(*centroids[ind_cur, :])
    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    poly = list(zip(vertices[ind_cur, :, 0], vertices[ind_cur, :, 1]))
    poly = plt.Polygon(poly, facecolor="orange", # type: ignore
                       edgecolor="none", alpha=0.5, zorder=-2)
    ax.add_patch(poly)
    plt.plot([pt_start.x, pt_end.x], [pt_start.y, pt_end.y],
             color="orange", linestyle="-", lw=0.8, zorder=-1)
    # -------------------------------------------------------------------------

    # Check intersection with triangle edges
    inters_found = False
    for i in range(3):
        ind_vt_1 = ind_loop[i]
        ind_vt_2 = ind_loop[i + 1]
        vertex_1 = Point(*vertices[ind_cur, ind_vt_1, :])
        vertex_2 = Point(*vertices[ind_cur, ind_vt_2, :])
        if intersect(pt_start, pt_end, vertex_1, vertex_2):
            inters_found = True
            break
    if inters_found == False:
        print("Break: Triangle containing point found")
        break
    i -= 1
    if i < 0:
        i = 2
    ind_prev = ind_cur
    ind_cur = neighbours[ind_cur][i]

    # Handle points outside of convex hull
    if ind_cur == -1:
        print("Break: Point is outside of convex hull")
        # ---------------------------------------------------------------------
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
        # https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        px = vertex_2.x - vertex_1.x
        py = vertex_2.y - vertex_1.y
        norm = px * px + py * py
        u =  (((pt_end.x - vertex_1.x) * px + (pt_end.y - vertex_1.y) * py) 
            / norm)
        if (u < 0) or (u > 1):
            # -----------------------------------------------------------------
            # Point is not perpendicular to intersected triangle edge
            # -----------------------------------------------------------------
            dist_sq_1 = (pt_end.x - vertex_1.x) ** 2 \
                + (pt_end.y - vertex_1.y) ** 2
            dist_sq_2 = (pt_end.x - vertex_2.x) ** 2 \
                + (pt_end.y - vertex_2.y) ** 2
            


            print("Warning: Special case: not yet implemented!")
            break
            # -----------------------------------------------------------------
        x_b = vertex_1.x + u * px
        y_b = vertex_1.y + u * py
        plt.plot([pt_end.x, x_b], [pt_end.y, y_b], color="black",
                 linestyle=":", lw=0.8, zorder=-1)
        plt.scatter(x_b, y_b, color="black", marker="*", s=50)
        # ---------------------------------------------------------------------
        break
# -----------------------------------------------------------------------------
plt.title(f"Number of iterations" + f": {count}", loc="left")
plt.axis((x_grid[0] - 0.5, x_grid[-1] + 0.5,
          y_grid[0] - 0.3, y_grid[-1] + 0.3))
plt.show()
