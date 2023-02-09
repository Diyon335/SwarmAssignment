import math
import random
import matplotlib.pyplot as plt
from matplotlib import animation


class Particle:
    """
    Class for a particle
    """

    def __init__(self, minimum, maximum):
        """
        Constructor for a particle

        :param minimum: The minimum value of the space in which the particle can lie
        :param maximum: The maximum value of the space in which the particle can lie
        """
        self.pos = (random.uniform(minimum, maximum), random.uniform(minimum, maximum))
        self.v = (random.uniform(minimum / 25, maximum / 25), random.uniform(minimum / 25, maximum / 25))
        self.f = 2 ** 32

        # Particle's best location
        self.sp_best = None
        # Global best location
        self.gb_best = None

        self.lowest_cost = 2 ** 32


def v_p(particle):
    """
    Calculates the new velocity vector based on the previous velocity, current position and the best personal and
    global position

    :param particle: Particle object for which the new velocity needs to be calculated
    :return: Returns a new tuple with the particle's new velocity in the x and y direction
    """

    x, y = particle.pos
    v_x, v_y = particle.v

    R = random.uniform(0, 1)

    new_v_x = (a_pso * v_x) + (b_pso * R * (particle.sp_best[0] - x)) + (c_pso * R * (particle.gb_best[0] - x))
    new_v_y = (a_pso * v_y) + (b_pso * R * (particle.sp_best[1] - y)) + (c_pso * R * (particle.gb_best[1] - y))

    # Caps the velocity
    if new_v_x > 5 / 25:
        new_v_x = 5 / 25

    elif new_v_x < -5 / 25:
        new_v_x = -5 / 25

    if new_v_y > 5 / 25:
        new_v_y = 5 / 25

    elif new_v_y < -5 / 25:
        new_v_y = -5 / 25

    return new_v_x, new_v_y


def s_p(particle):
    """
    Calculates the new position of the particle based on its previous position, current velocity and a time step (set
    to 1)

    :param particle: Particle object for which the new position needs to be calculated
    :return: Returns a new tuple with the particle's new position
    """

    x, y = particle.pos
    v_x, v_y = particle.v

    new_x = x + v_x
    new_y = y + v_y

    return new_x, new_y


def cost_rosenbrock(x, y):
    """
    The Rosenbrock cost function

    :param x: The x-coord of the particle
    :param y: The y-coord of the particle
    :return: Returns an integer indicating the cost
    """

    a = 0
    b = 100
    return (a - x) ** 2 + (b * (y - x ** 2) ** 2)


def cost_rastrigin(x, y, n=2):
    """
    The Rastrigin cost function

    :param x: The x-coord of the particle
    :param y: The y-coord of the particle
    :param n: Dimension of the space
    :return: Returns an integer indicating the cost
    """

    vector = [x, y]

    summation = 0
    for j in range(n):
        summation += vector[j] ** 2 - (10 * math.cos(2 * math.pi * vector[j] ** 2))

    return 10 * n + summation


def update_gb(particle_list, cost_function):
    """
    Updates the global best position of all particles

    :param particle_list: List of all the particles
    :param cost_function: The cost function being used
    :return: None
    """

    # Get a list of costs based on positions
    initial_costs = [cost_function(p.pos[0], p.pos[1]) for p in particle_list]

    # Find particle with the lowest cost
    index_lowest_cost = initial_costs.index(min(initial_costs))
    lowest_cost_particle = particle_list[index_lowest_cost]

    # Initialise all global best positions
    for p in particle_list:
        p.gb_best = lowest_cost_particle.pos


def plot_graphs():
    """
    Plots the graphs of each particle's position per test run

    :return: None
    """

    for n in range(tests):

        # Build lists of x and y coordinates for each particle at a specific test run (denoted by n)
        x = [particle_history[particle][n][0] for particle in particle_history]
        y = [particle_history[particle][n][1] for particle in particle_history]

        plt.cla()
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.scatter(x, y)

        # Plot the next graph after being delayed by 0.05 seconds
        plt.pause(0.05)


# Number of tests
tests = 1000

# Particle Swarm Optimisation (PSO) constants
b_pso, c_pso = 2, 2
a_pso = 0.9

# A constant subtracted from a_pso after each test that is run (0.9 - 0.4) range
d = 0.5 / tests

particle_history = {}

particles = [Particle(-5, 5) for x in range(20)]

if __name__ == '__main__':

    cost_function = cost_rosenbrock

    # Initialise an empty list to contain each particle's history
    for particle in particles:
        particle_history[particle] = []

    # Initialise all personal best positions to their initial position
    for particle in particles:
        particle.sp_best = particle.pos
        particle_history[particle].append(particle.pos)

    for i in range(tests):

        update_gb(particles, cost_function)

        for particle in particles:

            cost = cost_function(particle.pos[0], particle.pos[1])

            if cost < particle.lowest_cost:
                particle.lowest_cost = cost
                particle.sp_best = particle.pos

            x_update, y_update = s_p(particle)
            v_x_update, v_y_update = v_p(particle)

            particle.f = cost
            particle.pos = (x_update, y_update)
            particle_history[particle].append(particle.pos)
            particle.v = (v_x_update, v_y_update)

        a_pso -= d

    # Final positions and costs
    for particle in particles:
        print(particle.pos)
        print(particle.f)
        print()

    plot_graphs()
