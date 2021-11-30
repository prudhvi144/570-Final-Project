"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import numbers
import math
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical
    types
    """
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size



#3d rotational matrix

def rot2d(theta):
    """
    Change it  3D rotation matrix from the angle  theta according to (1).
    """
    rot_theta = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
    return rot_theta

#3D grid to plot 3d potential planner

class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate arbitrary
    functions on those points.
    """
    def __init__(self, xx_grid, yy_grid):
        """
        Stores the input arguments in attributes.
        """
        self.xx_grid = xx_grid
        self.yy_grid = yy_grid

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
        on each point defined by the grid.
        """
        dim_domain = [numel(self.xx_grid), numel(self.yy_grid)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_grid[idx_x]],
                                   [self.yy_grid[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])
        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """
        return np.meshgrid(self.xx_grid, self.yy_grid)



# Create the polygons and plot - total 6 poly gotms refer to the figure

class Polygon:
    """ Class for plotting, drawing, checking visibility and collision with polygons. """
    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    @property
    def nb_vertices(self):
        """ Number of vertices """
        return self.vertices.shape[1]

    def flip(self):
        """
        Reverse the order of the vertices (i.e., transform the polygon from filled in
        to hollow and viceversa).
        """
        self.vertices = np.fliplr(self.vertices)

    def plot(self, style):
        """
        Plot the polygon using Matplotlib.
        """
        if len(style) == 0:
            style = 'k'

        directions = np.diff(self.vertices_loop)
        # plt.fill(self.vertices[0, :],self.vertices[1, :],'silver')


        # plt.quiver(x,
        #            y,
        #            z,
        #            directions[0, :],
        #            directions[1, :],
        #            directions[1, :],
        #            color=style,
        #            width =0.003,
        #            angles='xyz',
        #            scale_units='xyz',
        #            scale=1.)
        fig = plt.figure()
        ax = Axes3D(fig)
        z = self.vertices[2, :]
        x =self.vertices[0, :]
        y = self.vertices[1, :]
        verts = [list(zip(x, y, z))]
        print(verts)
        ax.add_collection3d(Poly3DCollection(verts))
        plt.show()


#
#     def vertices_loop(self):
#         """
#         Returns self.vertices with the first vertex repeated at the end
#         """
#         return np.hstack((self.vertices, self.vertices[:, [0]]))
#
#     def is_filled(self):
#         """
#         Checks the ordering of the vertices, and returns whether the polygon is filled in or not.
#         """
#
#         # Iteratres over the columns of the 2D Matrix to perform the calculation
#         # sum((x_2 - x_1) * (y_2 + y_1))
#         # If the sum is negative, then the polygon is oriented counter-clockwise,
#         # clockwise otherwise.
#
#         num_cols = self.vertices.shape[1]
#         running_sum = 0
#
#         for i in range(num_cols - 1):
#             x_vals = self.vertices[0, :]
#             y_vals = self.vertices[1, :]
#
#             # modulus is for the last element to be compared with the first to close the shape
#             running_sum += (x_vals[(i+1) % num_cols] - x_vals[i]) * \
#                 (y_vals[i] + y_vals[(i+1) % num_cols])
#
#         return running_sum < 0
#
#     def is_self_occluded(self, idx_vertex, point):
#         """
#         Given the corner of a polygon, checks whether a given point is self-occluded or not by
#         that polygon (i.e., if it is ``inside'' the corner's cone or not). Points on boundary
#         (i.e., on one of the sides of the corner) are not considered self-occluded. Note that
#         to check self-occlusion, we just need a vertex index  idx_vertex. From this, one can
#         obtain the corresponding  vertex, and the  vertex_prev and  vertex_next that precede
#         and follow that vertex in the polygon.
#         """
#         vertex = self.vertices[:, [idx_vertex]]
#         vertex_next = self.vertices[:, [(idx_vertex + 1) % self.nb_vertices]]
#         vertex_prev = self.vertices[:, [(idx_vertex - 1) % self.nb_vertices]]
#
#         # The point is occluded if, measuring angles using p-vertex as the "zero angle",
#         # the angle for vertex_prev is smaller than the one for vertex_next
#         # Using the 'unsigned' angles means that we do not have to worry separately
#         # about negative angles
#         angle_p_prev = angle(vertex, point, vertex_prev, 'unsigned')
#         angle_p_next = angle(vertex, point, vertex_next, 'unsigned')
#
#         return angle_p_prev < angle_p_next
#
#     def is_visible(self, idx_vertex, test_points):
#         """
#         Checks whether a point p is visible from a vertex v of a polygon. In order to be visible,
#         two conditions need to be satisfied: enumerate  point p should not be self-occluded with
#         respect to the vertex v (see Polygon.is_self_occluded). The segment p--v should not collide
#         with any of the edges of the polygon (see Edge.is_collision).
#         """
#         nb_test_points = test_points.shape[1]
#         nb_vertices = self.vertices.shape[1]
#
#         # Initial default: all flags are True
#         flag_points = [True] * nb_test_points
#         vertex = self.vertices[:, [idx_vertex]]
#         for idx_point in range(0, nb_test_points):
#             point = test_points[:, [idx_point]]
#
#             # If it is self occluded, bail out
#             if self.is_self_occluded(idx_vertex, point):
#                 flag_points[idx_point] = False
#             else:
#                 # Build the vertex-point edge (it is the same for all other edges)
#                 edge_vertex_point = Edge(np.hstack([point, vertex]))
#                 # Then iterate over all edges in the polygon
#                 for idx_vertex_collision in range(0, self.nb_vertices):
#                     edge_vertex_vertex = Edge(self.vertices[:, [
#                         idx_vertex_collision,
#                         (idx_vertex_collision + 1) % nb_vertices
#                     ]])
#                     # The final result is the and of all the checks with individual edges
#                     flag_points[
#                         idx_point] &= not edge_vertex_point.is_collision(
#                             edge_vertex_vertex)
#
#                     # Early bail out after one collision
#                     if not flag_points[idx_point]:
#                         break
#
#         return flag_points
#
#     def is_collision(self, test_points):
#         """
#         Checks whether the a point is in collsion with a polygon (that is, inside for a filled in
#         polygon, and outside for a hollow polygon). In the context of this homework, this function
#         is best implemented using Polygon.is_visible.
#         """
#
#         flag_points = [False] * test_points.shape[1]
#         # We iterate over the polygon vertices, and process all the test points in parallel
#         for idx_vertex in range(0, self.nb_vertices):
#             flag_points_vertex = self.is_visible(idx_vertex, test_points)
#             # Accumulate the new flags with the previous ones
#             flag_points = [
#                 flag_prev or flag_new
#                 for flag_prev, flag_new in zip(flag_points, flag_points_vertex)
#             ]
#         flag_points = [not flag for flag in flag_points]
#         return flag_points
# class Edge:
#     """ Class for storing edges and checking collisions among them. """
#     def __init__(self, vertices):
#         """
#         Save the input coordinates to the internal attribute  vertices.
#         """
#         self.vertices = vertices
#
#     @property
#     def direction(self):
#         """ Difference between tip and base """
#         return self.vertices[:, [1]] - self.vertices[:, [0]]
#
#     @property
#     def base(self):
#         """ Coordinates of the first vertex"""
#         return self.vertices[:, [0]]
#
#     def is_collision(self, edge):
#         """
#         Returns  True if the two edges intersect.  Note: if the two edges overlap but are colinear,
#         or they overlap only at a single endpoint, they are not considered as intersecting (i.e.,
#         in these cases the function returns  False). If one of the two edges has zero length, the
#         function should always return the result that edges are non-intersecting.
#         """
#
#         # Write the lines from the two edges as x_i(t_i)=edge_base+edge.direction*t_i
#         # Then finds the parameters for the intersection by solving the linear system obtained from
#         # x_1(t_1)=x_2(t_2)
#
#         # Tolerance for cases involving parallel lines and endpoints
#         tol = 1e-6
#
#         # The matrix of the linear system
#         a_directions = np.hstack([self.direction, -edge.direction])
#         if abs(np.linalg.det(a_directions)) < tol:
#             # Lines are practically parallel
#             return False
#         # The vector of the linear system
#         b_bases = np.hstack([edge.base - self.base])
#
#         # Solve the linear system
#         t_param = np.linalg.solve(a_directions, b_bases)
#         t_self = t_param[0, 0]
#         t_other = t_param[1, 0]
#
#         # Check that collision point is strictly between endpoints of each edge
#         flag_collision = tol < t_self < 1.0 - tol and tol < t_other < 1.0 - tol
#
#         return flag_collision
#
#
# def angle(vertex0, vertex1, vertex2, angle_type='signed'):
#     """
#     Compute the angle between two edges  vertex0-vertex1 and  vertex0-vertex2 having an endpoint in
#     common. The angle is computed by starting from the edge  vertex0-- vertex1, and then
#     ``walking'' in a counterclockwise manner until the edge  vertex0-vertex2 is found.
#     The angle is computed by starting from the vertex0-vertex1 edge, and then “walking” in a
#     counterclockwise manner until the is found.
#     """
#     # tolerance to check for coincident points
#     tol = 2.22e-16
#
#     # compute vectors corresponding to the two edges, and normalize
#     vec1 = vertex1 - vertex0
#     vec2 = vertex2 - vertex0
#
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     if norm_vec1 < tol or norm_vec2 < tol:
#         # vertex1 or vertex2 coincides with vertex0, abort
#         edge_angle = math.nan
#         return edge_angle
#
#     vec1 = vec1 / norm_vec1
#     vec2 = vec2 / norm_vec2
#
#     # Transform vec1 and vec2 into flat 3-D vectors,
#     # so that they can be used with np.inner and np.cross
#     vec1flat = np.vstack([vec1, 0]).flatten()
#     vec2flat = np.vstack([vec2, 0]).flatten()
#
#     c_angle = np.inner(vec1flat, vec2flat)
#     s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))
#
#     edge_angle = math.atan2(s_angle, c_angle)
#
#     angle_type = angle_type.lower()
#     if angle_type == 'signed':
#         # nothing to do
#         pass
#     elif angle_type == 'unsigned':
#         edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
#     else:
#         raise ValueError('Invalid argument angle_type')
#
#     return edge_angle

if __name__ == '__main__':


 t = [0,45]
 t1 = Torus(t)
 t1.plot_charts()
