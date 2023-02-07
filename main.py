import math
import random


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
        self.v = (random.uniform(minimum/25, maximum/25), random.uniform(minimum/25, maximum/25))
        self.f = 2**32

        # Particle's best location
        self.sp_best = None
        # Global best location
        self.gb_best = None


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


def cost_rosenbrock(particle):
    """
    The Rosenbrock cost function

    :param particle: Particle object
    :return: Returns an integer indicating the cost
    """

    a = 0
    b = 100

    return (a - particle.pos[0])**2 + (b * (particle.pos[1] - particle.pos[0]**2)**2)


def cost_rastrigin(particle, n=2):
    """
    The Rastrigin cost function

    :param particle: Particle object
    :param n: Dimension of the space
    :return: Returns an integer indicating the cost
    """

    summation = 0
    for j in range(n):

        summation += particle.pos[0]**2 - (10 * math.cos(2 * math.pi * particle.pos[0]**2))

    return 10*n + summation


def update_gb(particles, cost_function):
    """
    Updates the global best position of all particles

    :param particles: List of all the particles
    :param cost_function: The cost function being used
    :return: None
    """

    # Get a list of costs based on positions
    initial_costs = [cost_function(particle) for particle in particles]

    # Find particle with the lowest cost
    index_lowest_cost = initial_costs.index(min(initial_costs))
    lowest_cost_particle = particles[index_lowest_cost]

    # Initialise all global best positions
    for particle in particles:
        particle.gb_best = lowest_cost_particle.pos

# Number of tests
tests = 1000

# Particle Swarm Optimisation (PSO) constants
b_pso, c_pso = 2, 2
a_pso = 0.9

# A constant subtracted from a_pso after each test that is run
d = 0.5/tests


if __name__ == '__main__':

    particles = [Particle(-3, 3) for x in range(5)]

    cost_function = cost_rastrigin

    # Compute the initial best global position
    update_gb(particles, cost_function)

    # TODO Not sure about this
    # Initialise all personal best positions to the initial position
    for particle in particles:
        particle.sp_best = particle.pos

    for i in range(tests):

        for particle in particles:

            x_update, y_update = s_p(particle)
            v_x_update, v_y_update = v_p(particle)

            cost = cost_function(particle)

            if cost < particle.f:

                particle.sp_best = (x_update, y_update)

            particle.f = cost
            particle.pos = (x_update, y_update)
            particle.v = (v_x_update, v_y_update)

        update_gb(particles, cost_function)

        a_pso -= d

    for particle in particles:
        print(particle.f)
