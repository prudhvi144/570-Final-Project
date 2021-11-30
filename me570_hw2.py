"""
Main file for ME570 HW2
"""

import math
import time
from scipy import io as scio
import numpy as np
import matplotlib.pyplot as plt
import me570_geometry
import me570_robot
import time
# import robot

from mpl_toolkits.mplot3d import Axes3D

def twolink_plot_collision_test():
    """
    This function generates 30 random configurations, loads the  points variable from the file
!70!DarkSeaGreen2 twolink_testData.mat (provided with the homework), and then display the results
using  twolink_plotCollision to plot the manipulator in red if it is in collision, and green
otherwise.
    """
    nb_configurations = 10
    two_link = me570_robot.TwoLink()
    theta_random = 2 * math.pi * np.random.rand(2, nb_configurations)
    test_data = scio.loadmat('twolink_testData.mat')
    obstacle_points = test_data['obstaclePoints']
    fig = plt.figure(figsize=(9, 3))
    ax = fig.gca()
    ax.set_aspect('equal')
    plt.plot(obstacle_points[0, :], obstacle_points[1, :], 'b*')
    for i_theta in range(0, nb_configurations):
        theta = theta_random[:, i_theta:i_theta + 1]
        flag_theta= two_link.plot_collision(theta, obstacle_points)
    plt.show()

def grid_eval_example():

    """ Example of the use of Grid.mesh and Grid.eval functions"""
    fun = lambda x: math.sin(x[0])
    example_grid = me570_geometry.Grid(np.linspace(-3, 3), np.linspace(-3, 3))
    fun_eval = example_grid.eval(fun)
    [xx_grid, yy_grid] = example_grid.mesh()
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    print (fun_eval.shape)
    axis.plot_surface(xx_grid, yy_grid, fun_eval)

    plt.show()


def torus_twolink_plot_jacobian():

    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca()
    ax.set_aspect('equal')
    # a_line = np.array([[(3 / 4) * math.pi], [0]])
    # a_line = np.array([[(3 / 4) * math.pi], [3 / 4 * math.pi]])
    a_line = np.array([[(-3 / 4) * math.pi], [3 / 4 * math.pi]])
    # a_line = np.array([[0],[(3 / 4)*(math.pi)]])
    b_line = np.array([[-1], [-1]])
    N = 10
    thetaPoints =  me570_geometry.line_linspace(a_line, b_line, 0, 1, N)
    p = me570_robot.TwoLink()


    for i in range(10):
        the = thetaPoints[:,i]
        p.plot(thetaPoints[:,i], 'g')
        v = p.jacobian(thetaPoints[:,i], a_line)
        pos,_,_ = p.kinematic_map(the)
        print (v[0],v[1])
        print(pos,v)
        plt.plot(pos[0],pos[1],'bs')
        # plt.quiver(pos[0],pos[1],v[0],v[1],color='r',width=0.003,angles='xy',scale_units='xy',scale=5)
        plt.xlim(-15, 5)
        plt.ylim(-15, 10)
    plt.show()

















    # the = [0,0]
    # print (the[0])
    # a = [0,1]
    # p.plot(the, 'g')
    # v = p.jacobian(the, a)
    # pos,_,_ = p.kinematic_map(the)
    # print (v[0],v[1])
    # print(pos,v)
    # plt.plot(pos[0],pos[1],'bs')
    # plt.quiver(pos[0],pos[1],v[0],v[1],color='r',width=0.003,angles='xy',scale_units='xy',scale=5)
    # plt.xlim(-10, 15)
    # plt.ylim(-10, 10)
    # the = [0,math.pi]
    # print (the[0])
    # a = [0,1]
    # p.plot(the, 'g')
    # v = p.jacobian(the, a)
    # pos,_,_ = p.kinematic_map(the)
    # print (v[0],v[1])
    # print(pos,v)
    # plt.plot(pos[0],pos[1],'bs')
    # plt.quiver(pos[0],pos[1],v[0],v[1],color='r',width=0.003,angles='xy',scale_units='xy',scale=5)
    # plt.xlim(-10, 15)
    # plt.ylim(-10, 10)
    # the = [0,math.pi/2]
    # print (the[0])
    # a = [0,1]
    # p.plot(the, 'g')
    # v = p.jacobian(the, a)
    # pos,_,_ = p.kinematic_map(the)
    # print (v[0],v[1])
    # print(pos,v)
    # plt.plot(pos[0],pos[1],'bs')
    # plt.quiver(pos[0],pos[1],v[0],v[1],color='r',width=0.003,angles='xy',scale_units='xy',scale=5)
    # plt.xlim(-10, 15)
    # plt.ylim(-10, 10)
    # the = [0,math.pi/2]
    # print (the[0])
    # a = [0,1]
    # p.plot(the, 'g')
    # v = p.jacobian(the, a)
    # pos,_,_ = p.kinematic_map(the)
    # print (v[0],v[1])
    # print(pos,v)
    # plt.plot(pos[0],pos[1],'bs')
    # plt.quiver(pos[0],pos[1],v[0],v[1],color='r',width=0.003,angles='xy',scale_units='xy',scale=5)
    # plt.xlim(-10, 15)
    # plt.ylim(-10, 10)
    # plt.show()
twolink_plot_collision_test()
