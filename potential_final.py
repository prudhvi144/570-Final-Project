"""
Classes to define potential and potential planner for the sphere world
"""

import numpy as np
import geometry_final as gm
from matplotlib import pyplot as plt
from scipy import io as scio


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """
        data = scio.loadmat('sphereWorld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = np.asscalar(sphere_args[1])
            sphere_args[2] = np.asscalar(sphere_args[2])
            self.world.append(gm.Sphere(*sphere_args))

        a = self.world
        print(a)
        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def _workspace_meshgrid(self):
        x = np.arange(-10 - .1, 10 + .1, .05)
        y = np.arange(-10 - .1, 10 + .1, .05)

        xx, yy = np.meshgrid(x, y)

        return xx, yy

    def plot(self):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a  * marker at the goal location.
        """
        plt.subplot(1, 2, 1)
        for sphere in self.world:
            sphere.plot('r')

        plt.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])


class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """
        Evaluate the repulsive potential from  sphere at the location x= x_eval. The function returns
    the repulsive potential as given by      (  eq:repulsive  ).
        """

        p1 = gm.Sphere(self.sphere.center, self.sphere.radius,
                       self.sphere.distance_influence)
        x_eval = x_eval.reshape((-1, 1))
        dq = np.array(p1.distance(x_eval))
        rr = np.array(p1.distance_influence)

        if dq <= rr:
            if dq <= 0.1:
                dq = 0.1
            u_rep = 0.5 * (1.0 / dq - 1.0 / rr)**2
        else:
            u_rep = 0.0

        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by      (  eq:repulsive-gradient
    ).
        """
        p1 = gm.Sphere(self.sphere.center, self.sphere.radius,
                       self.sphere.distance_influence)
        x_eval = x_eval.reshape((-1, 1))
        dq = np.array(p1.distance(x_eval))
        dg = np.array(p1.distance_grad(x_eval)).reshape(2, 1).T
        rr = np.array(p1.distance_influence)
        grad_u_rep = np.zeros(2).reshape(2, 1).T

        if (dq < rr and dq > 0):
            grad_u_rep = -((1.0 / dq - 1.0 / rr) * (1 / (dq**2)) * dg)
        else:
            grad_u_rep = grad_u_rep
        # print(grad_u_rep)
        return grad_u_rep


class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        ee = x_eval
        if self.potential['shape'] == 'conic':
            p = 1
        else:
            p = 2
        ppo = np.linalg.norm(x_eval - self.potential['x_goal'][:, 1])
        dq = (np.linalg.norm(x_eval - self.potential['x_goal'][:, 1]))**p
        u_attr = dq
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient is
    given by the formula If  potential['shape'] is equal to  'conic', use p=1; if it is equal to
    'quadratic', use p=2.
        """
        if self.potential['shape'] == 'conic':
            p = 1
        else:
            p = 2
        grad_u_attr = p * (np.linalg.norm(x_eval -
                                          self.potential['x_goal'][:, 1])**
                           (p - 2)) * (x_eval - self.potential['x_goal'][:, 1])

        return grad_u_attr


class navigation:
    """ Repulsive potential for a sphere """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential
        self.world = world
        self._k = 16

    def _evaluate_gamma(self, x_eval):

        gamma = np.linalg.norm(x_eval - potential['x_goal'][:, 1])**(2 *
                                                                     self._k)

        return gamma

    @staticmethod
    def _evaluate_beta_i(x_eval, sphere):
        return np.linalg.norm(x_eval - sphere.center)**2 - sphere.radius**2

    def _evaluate_beta(self, x_eval):
        beta = -self.world[0].radius**2 - np.linalg.norm(
            x_eval - self.world[0].center)**2
        for i in range(1, 3):
            beta *= self._evaluate_beta_i(x_eval, self.world[i])

        return beta

    def _evaluate_alpha(self, x_eval):
        gamma = self._evaluate_gamma(x_eval)
        beta = self._evaluate_beta(x_eval)
        if abs(beta) < 0.00001:
            alpha = 1e15
        else:
            alpha = gamma / beta

        return alpha

    def _evaluate_phi(self, x_eval):
        alpha = self._evaluate_alpha(x_eval)
        if alpha < 0:
            return 1
        else:
            phi = (alpha / (1 + alpha))**(1 / self._k)

        return phi

    def evaluate(self, x_eval):
        return self._evaluate_phi(x_eval)

    def _evaluate_grad_gamma(self, x_eval):
        # grad_gamma = 2 * self._k * distance(q, self._goal) ** (2 * self._k - 1) * (q - self._goal) / distance(q,
        #                                                                                                       self._goal)
        grad_gamma = (np.linalg.norm(x_eval - self.potential['x_goal'][:, 1])
                      **(1 - 2)) * (x_eval - self.potential['x_goal'][:, 1])

        return grad_gamma

    def _evaluate_grad_beta(self, x_eval):
        beta_0 = -self.world[0].radius**2 - np.linalg.norm(
            x_eval - np.squeeze(self.world[0].center))**2
        grad_beta_0 = -2 * (x_eval - np.squeeze(self.world[0].center))

        betas = [beta_0]
        grad_betas = [grad_beta_0]

        for i in range(1, 3):
            beta_i = self._evaluate_beta_i(x_eval, self.world[i])
            grad_beta_i = 2 * (x_eval - np.squeeze(self.world[i].center))

            betas.append(beta_i)
            grad_betas.append(grad_beta_i)

        grad_beta = np.zeros(2)
        n = len(betas)
        for i in range(n):
            product = grad_betas[i]
            for j in range(n):
                if j != i:
                    product *= betas[j]

            grad_beta += product

        return grad_beta

    def _evaluate_grad_alpha(self, q: np.ndarray) -> np.ndarray:
        gamma = self._evaluate_gamma(q)
        beta = self._evaluate_beta(q)
        grad_gamma = self._evaluate_grad_gamma(q)
        grad_beta = self._evaluate_grad_beta(q)
        grad_alpha = (grad_gamma * beta - gamma * grad_beta) / beta**2

        return grad_alpha

    def _evaluate_grad_phi(self, q: np.ndarray) -> np.ndarray:
        alpha = self._evaluate_alpha(q)
        grad_alpha = self._evaluate_grad_alpha(q)
        grad_phi = (1 / self._k) * (alpha / (1 + alpha))**(
            (1 - self._k) / self._k) * (1 / (1 + alpha)**2) * grad_alpha

        return grad_phi

    def evaluate_gradient(self, q: np.ndarray) -> np.ndarray:
        return self._evaluate_grad_phi(q)


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential

    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """

        # potRepTot = []
        # n = len(self.world)
        # for i in range(n):
        #     p3 = RepulsiveSphere(self.world[i])
        #     potRepTot = potRepTot + p3.eval(x_eval)
        # xx= Attractive(self.potential)
        # u_eval = xx.eval(x_eval) + self.potential['repulsiveWeight'] * potRepTot
        # return u_eval

        # u_attr = np.zeros(1)
        # for sphere in self.world:
        #     u_attr =u_attr + Attractive(self.potential).eval(x_eval)

        u_rep = np.zeros(1)
        for sphere in self.world:

            u_rep = u_rep + RepulsiveSphere(sphere).eval(x_eval)

        u_eval = Attractive(self.potential).eval(
            x_eval) + self.potential['repulsiveWeight'] * u_rep
        return u_eval

    def grad(self, x_eval):
        """
        Compute the gradient of the total potential,  U= U_ attr+    _i U_ rep,i, where   is given by
    the variable  potential.repulsiveWeight
        """

        grad_u_rep = np.zeros(2)
        for sphere in self.world:

            grad_u_rep = grad_u_rep + RepulsiveSphere(sphere).grad(x_eval)

        grad_u_eval = np.add(
            Attractive(self.potential).grad(x_eval),
            self.potential['repulsiveWeight'] * grad_u_rep)
        return grad_u_eval


class Planner:
    """  """
    def run(self, x_start, planned_parameters):
        """
        This function uses a given function ( planner_parameters['control']) to implement a generic
    potential-based planner with step size  planner_parameters['epsilon'], and evaluates the cost
    along the returned path. The planner must stop when either the number of steps given by
    planner_parameters['nb_steps'] is reached, or when the norm of the vector given by
    planner_parameters['control'] is less than 5 10^-3 (equivalently,  5e-3).
        """
        p = SphereWorld()
        potential = {
            'x_goal': p.x_goal,
            'shape': 'conic',
            'repulsiveWeight': 1
        }
        x_goal = np.array(potential['x_goal'])
        nb_steps = planned_parameters['nb_steps']
        epsilon = planned_parameters['epsilon']
        control = planned_parameters['control']
        repulsive_weight = potential['repulsiveWeight']
        shape = potential['shape']

        p = SphereWorld()
        total = Total(p.world, potential)

        x_eval = x_start
        x_path = x_start.T
        # u_path = Attractive(potential).eval(x_eval)
        u_path = planned_parameters['U'](x_eval)
        x_path = x_path
        b = len(x_start)

        for i in range(nb_steps):
            ug_eval = control(x_eval)

            x_eval = x_eval + epsilon * ug_eval

            # print (x_eval)

            x_path = np.vstack((x_path, x_eval))
            u_eval = planned_parameters['U'](x_path[i + 1])
            u_path = np.vstack((u_path, u_eval))

            if np.linalg.norm(u_eval) < 0.0005:
                break
        return x_path, u_path

    def run_plot(self):
        """
        This function performs the following steps:
     - Loads the problem data from the file !70!DarkSeaGreen2 sphereworld.mat.
     - For each goal location in  world.xGoal:
     - Uses the function Sphereworld.plot to plot the world in a first figure.
     - Sets  planner_parameters['U'] to the negative of  Total.grad.
     - it:grad-handle Calls the function Potential.planner with the problem data and the input
    arguments. The function needs to be called five times, using each one of the initial locations
    given in  x_start (also provided in !70!DarkSeaGreen2 sphereworld.mat).
     - it:plot-plan After each call, plot the resulting trajectory superimposed to the world in the
    first subplot; in a second subplot, show  u_path (using the same color and using the  semilogy
    command).
        """
        data = scio.loadmat('sphereWorld.mat')

        self.world = []

        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = np.asscalar(sphere_args[1])
            sphere_args[2] = np.asscalar(sphere_args[2])
            self.world.append(gm.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']
        n = len(self.x_goal)
        m = 1
        p = SphereWorld()
        p.plot()

        for i in range(m):

            p1 = Planner()
            x_path, u_path = p1.run(p.x_start[:, i], planned_parameters)
            x_path = x_path.T
            print(x_path)
            plt.subplot(1, 2, 1)
            plt.plot(x_path[0], x_path[1])
            plt.subplot(1, 2, 2)
            plt.plot(u_path)

        # plt.xlim([-11, 11])
        # plt.ylim([-11, 11])
        plt.show()


if __name__ == '__main__':
    p = SphereWorld()
    potential = {'x_goal': p.x_goal, 'shape': 'conic', 'repulsiveWeight': 0.05}

    p2 = Total(p.world, potential)
    # print(p.world[0])
    planned_parameters = {
        'nb_steps': 1900,
        'epsilon': 0.01,
        'U': lambda x: p2.eval(x),
        'control': lambda x: -p2.grad(x)
    }

    # for i in len(potential['x_goal']):
    p1 = Planner()
    p1.run_plot()

    sphere = gm.Sphere(2 * np.ones((2, 1)), -2, 3)

    for sphere in (p.world):
        sphere = gm.Sphere(sphere.center, sphere.radius,
                           sphere.distance_influence)
        f_handle = lambda x: p2.eval(x.T[0])
        gm.field_plot_threshold(f_handle, 10)

