from math import pi
from operator import pos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Linkage import Linkage
from Plate import Plate
from Neckbrace import Neckbrace
import potential_final as pf
import geometry_final as gm
from scipy import io as scio


def get_path():
    p = pf.SphereWorld()
    data = scio.loadmat('sphereWorld6.mat')
    world = []
    for sphere_args in np.reshape(data['world'], (-1,)):
        sphere_args[1] = np.asscalar(sphere_args[1])
        sphere_args[2] = np.asscalar(sphere_args[2])
        world.append(gm.Sphere(*sphere_args))

    x_goal = data['xGoal']
    x_start = data['xStart']
    potential = {'x_goal': x_goal, 'shape': 'conic', 'repulsiveWeight': 0.05}
    p2 = pf.Total(p.world, potential)
    planned_parameters = {
        'nb_steps': 1600,
        'epsilon': 0.01,
        'U': lambda x: p2.eval(x),
        'control': lambda x: -p2.grad(x)
    }
    p1 = pf.Planner()
    xx = x_start[:, 0]
    x_path, u_path = p1.run(x_start[:, 0], planned_parameters)
    print (x_path)
    m = len(x_path)
    z = 15*np.ones(m)
    x_path = x_path.T
    x = x_path[0]
    y = x_path[1]

    # for sphere in (p.world):
    #     sphere = gm.Sphere(sphere.center, sphere.radius,
    #                        sphere.distance_influence)
    #     f_handle = lambda x: p2.eval(x.T[0])
    #     gm.field_plot_threshold(f_handle, 100)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.set_xlim3d(xmin=-7, xmax=7)
    # ax.set_xlabel("X", fontsize=20)
    # ax.set_ylim3d(ymin=-7, ymax=7)
    # ax.set_ylabel("Y", fontsize=20)
    # ax.set_zlim3d(zmin=0, zmax=14)
    # ax.set_zlabel("Z", fontsize=20)
    # ax.plot3D(x, y, z, 'gray')

    return (x,y,z)




def main():
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(xmin=-7, xmax=7)
    ax.set_xlabel("X", fontsize=20)
    ax.set_ylim3d(ymin=-7, ymax=7)
    ax.set_ylabel("Y", fontsize=20)
    ax.set_zlim3d(zmin=0, zmax=14)
    ax.set_zlabel("Z", fontsize=20)
    x, y, z = get_path()
    # len_p=len(x)
    # x1 = 0 * np.ones(len_p)
    # y1 = 0 * np.ones(len_p)
    # z1 = 4 * np.ones(len_p)

    # alpha =np.arccos(np.dot(x,x1)/(np.linalg.norm(x)*np.linalg.norm(x1)))
    #
    #
    # beta =  np.arccos(np.dot(y,y1)/(np.linalg.norm(y)*np.linalg.norm(y1)))

    alpha =  np.arctan(x/z)[0::50]
    beta =   np.arctan(y/z)[0::50]



    print (alpha)

    # Neckbrace robot with ring radius of 5
    neckbrace = Neckbrace(4, 4.5,
                          [pi / 6, 5 * pi / 6, 7 * pi / 6, 11 * pi / 6])

    div_pi_by = 18
    step_1 = np.linspace(pi / div_pi_by, 0, 5, endpoint=False)
    step_2 = np.linspace(0, -pi / div_pi_by, 5, endpoint=False)
    step_3 = np.linspace(-pi / div_pi_by, 0, 5, endpoint=False)
    step_4 = np.linspace(0, pi / div_pi_by, 5)

    # positions = np.hstack((step_1, step_2, step_3, step_4))
    positions = np.vstack((beta,alpha))

    azimuth = -84
    elevation = 45
    neckbrace.animate(positions, "normal_first.gif", azimuth, elevation, x,-y ,z)
    neckbrace.animate1(positions, "normal_first.gif", azimuth, elevation, x,-y ,z)


if __name__ == "__main__":
    get_path()
    main()
    plt.show()

