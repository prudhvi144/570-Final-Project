"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import me570_geometry as gm
import numpy as np
import matplotlib.pyplot as plt
import math


# class for the top ring plus 4 links
class TwoLink:
    def __init__(self):
        add_y_reflection = lambda vertices: np.hstack(
            [vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
        self.vertices1 = np.array([[0, 5, 5, 0], [1, 1, -1, -1], [1, 1, 1, 1]])
        # self.vertices1 = add_y_reflection(vertices1)
        v = self.vertices1
        self.vertices2 = np.array([[0, 6], [0, 0], [1, 1]])

        # self.vertices2 = add_y_reflection(vertices2)
        v2 = self.vertices2
        self.polygons = (gm.Polygon(self.vertices1),
                         gm.Polygon(self.vertices2))

    """ This class was introduced in a previous homework. """

    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
    transformed according to  _1, _2.
        """
        # returns the coordinate of the end effector
        WR1 = np.array(gm.rot2d(theta[0]))
        Rb2 = np.array(gm.rot2d(theta[1]))
        temp_end = WR1.dot([[5], [0]])
        temp2_end = Rb2.dot([[5], [0]])
        vertex_effector_transf = WR1.dot(temp2_end) + temp_end

        # returns vertices of the links
        polygon1_transf = WR1.dot(self.vertices1)
        temp = WR1.dot([[5], [0]])
        temp2 = Rb2.dot(self.vertices2)
        polygon2_transf = WR1.dot(temp2) + temp

        return vertex_effector_transf, polygon1_transf, polygon2_transf

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        [vertex_effector_transf, polygon1_transf,
         polygon2_transf] = self.kinematic_map(theta)
        # fig = plt.figure()
        # ax = fig.gca()
        # ax.set_aspect('equal')
        p1 = gm.Polygon(polygon1_transf)
        p2 = gm.Polygon(polygon2_transf)

        # p1.plot(color)
        p2.plot(color)
        p1.plot(color)

        plt.show()

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """
        [vertex_effector_transf, polygon1_transf,
         polygon2_transf] = self.kinematic_map(theta)
        p1 = gm.Polygon(polygon1_transf)
        p2 = gm.Polygon(polygon2_transf)
        flag_theta = []

        if any(p1.is_collision(points)) or any(p2.is_collision(points)):
            flag_theta.append('False')
        else:
            flag_theta.append('True')
        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """

        pp = self.is_collision(theta, points)
        print(pp[0])
        if pp[0] == 'True':
            print('in')
            self.plot(theta, 'g')
        else:
            print('out')
            self.plot(theta, 'r')

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """

        vertex_effector_dot_1 = -5 * (math.sin(theta[0])) * (theta_dot[0]) - (
            5 * (math.cos(theta[0])) * (math.sin(theta[1])) *
            (theta_dot[0])) - (
                5 * (math.cos(theta[1])) * (math.sin(theta[0])) *
                (theta_dot[0])) - (5 * (math.cos(theta[0])) *
                                   (math.sin(theta[1])) *
                                   (theta_dot[1])) - (5 *
                                                      (math.cos(theta[1])) *
                                                      (math.sin(theta[0])) *
                                                      (theta_dot[1]))
        vertex_effector_dot_2 = 5 * (math.cos(theta[0])) * (theta_dot[0]) + (
            5 * (math.cos(theta[0])) * (math.cos(theta[1])) *
            (theta_dot[0])) + (
                5 * (math.cos(theta[0])) * (math.cos(theta[1])) *
                (theta_dot[1])) - (5 * (math.sin(theta[0])) *
                                   (math.sin(theta[1])) *
                                   (theta_dot[0])) - (5 *
                                                      (math.sin(theta[0])) *
                                                      (math.sin(theta[1])) *
                                                      (theta_dot[1]))
        vertex_effector_dot = [vertex_effector_dot_1, vertex_effector_dot_2]

        return vertex_effector_dot


p = TwoLink()
the = [0, 0]
print(the[0])
a = [0, 1]
p1 = gm.Polygon(p.vertices1)
p2 = gm.Polygon(p.vertices2)
color = 'g'
# p1.plot(color)
p2.plot(color)
p1.plot(color)
# plt.show()