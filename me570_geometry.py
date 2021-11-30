"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import numbers
from math import sin, cos
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def rot3d(alpha, beta, gamma):
    """
    Rotates a point in 3D space where:
        alpha determines x_axis rotation
        beta determines y_axis rotation
        gamma determines z_axis rotation
    """
    sin_alpha = sin(alpha)
    sin_beta = sin(beta)
    sin_gamma = sin(gamma)

    cos_alpha = cos(alpha)
    cos_beta = cos(beta)
    cos_gamma = cos(gamma)

    return np.array(
        [[
            cos_gamma * cos_beta,
            cos_gamma * sin_beta * sin_alpha - sin_gamma * cos_alpha,
            cos_gamma * sin_beta * cos_alpha + sin_gamma * sin_alpha
        ],
         [
             sin_gamma * cos_beta,
             sin_gamma * sin_beta * sin_alpha + cos_gamma * cos_alpha,
             sin_gamma * sin_beta * cos_alpha - cos_gamma * sin_alpha
         ], [-sin_beta, cos_beta * sin_alpha, cos_beta * cos_alpha]])


#3D grid to plot 3d potential planner

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

        fig = plt.figure()
        ax = Axes3D(fig)
        z = self.vertices[2, :]
        x = self.vertices[0, :]
        y = self.vertices[1, :]
        verts = [list(zip(x, y, z))]
        print(verts)
        ax.add_collection3d(Poly3DCollection(verts))
        plt.show()