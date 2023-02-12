import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class Particle:
    """
    Class for a particle
    """

    def __init__(self, id, minimum, maximum):
        """
        Constructor for a particle

        :param minimum: The minimum value of the space in which the particle can lie
        :param maximum: The maximum value of the space in which the particle can lie
        """
        self.id = id
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

    R = random.uniform(0, 1)

    pos = np.array(particle.pos)
    v = np.array(particle.v)
    sp_best = np.array(particle.sp_best)
    gb_best = np.array(particle.gb_best)

    new_v = (a_pso * v) + (b_pso * R * (sp_best - pos)) + (c_pso * R * (gb_best - pos))

    # Caps the velocity
    norm = np.linalg.norm(new_v)
    if norm > v_max:
        new_v = (new_v / norm) * v_max

    return new_v[0], new_v[1]


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
        summation += vector[j] ** 2 - (10 * np.cos(2 * np.pi * vector[j] ** 2))

    return (10 * n) + summation


def update_gb(particle_list, cost_function, neighborhood = "global"):
    """
    Updates the global best position of all particles

    :param particle_list: List of all the particles
    :param cost_function: The cost function being used
    :return: None
    """

    # Get a list of costs based on positions
    initial_costs = [cost_function(p.pos[0], p.pos[1]) for p in particle_list]

    if neighborhood == "global":
        # Find particle with the lowest cost
        index_lowest_cost = initial_costs.index(min(initial_costs))
        lowest_cost_particle = particle_list[index_lowest_cost]

        # Initialise all global best positions
        for p in particle_list:
            p.gb_best = lowest_cost_particle.pos

    elif neighborhood == "social":

        i = 0
        while i < len(initial_costs):
            index_lowest_cost = initial_costs[i:i+5].index(min(initial_costs[i:i+5]))
            lowest_cost_particle = particle_list[index_lowest_cost]

            for p in particle_list[i:i+5]:
                p.gb_best = lowest_cost_particle.pos

            i += 5

    elif neighborhood == "geographical":

        for p in particle_list:
            neighborhood = []
            for other_p in particle_list:
                distance = math.dist(p.pos, other_p.pos)
                neighborhood.append((distance, other_p.pos, initial_costs[particle_list.index(other_p)]))
            neighborhood.sort()
            neighborhood = neighborhood[:4]
            p.gb_best = min(neighborhood, key = lambda x: x[2])[1]


def plot_graphs(cost_function):
    """
    Plots the graphs of each particle's position per test run

    :return: None
    """
    fig, ax = plt.subplots()

    n = 100
    X = np.linspace(x_range[0], x_range[1], n)
    Y = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(X, Y)

    Z = cost_function(X, Y)

    pcm = plt.pcolor(X, Y, Z, norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()), cmap='jet', shading='auto')
    fig.colorbar(pcm, extend='max')

    prev_x = [particle_history[particle][0][0] for particle in particle_history]
    prev_y = [particle_history[particle][0][1] for particle in particle_history]

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    (ln,) = ax.plot(prev_x, prev_y, 'bo', animated=True)

    plt.show(block=False)
    plt.pause(2)

    bg = fig.canvas.copy_from_bbox(fig.bbox)

    for n in range(tests - 1):

        ax.set_title(f"Particle convergence using PSO with {len(particle_history)} particles (iteration: {n+2})")

        text = f"a = {round(parameter_history[n+1][0], 2)}\n " \
               f"b = {round(parameter_history[n+1][1], 2)}\n " \
               f"c = {round(parameter_history[n+1][2], 2)}\n"

        ax.text(3, 3, text, fontsize=10)

        # Build lists of x and y coordinates for each particle at a specific test run (denoted by n)
        x = [particle_history[particle][n+1][0] for particle in particle_history]
        y = [particle_history[particle][n+1][1] for particle in particle_history]
        # reset the background back in the canvas state, screen unchanged
        fig.canvas.restore_region(bg)
        # update the artist, neither the canvas state nor the screen have changed
        ln.set_xdata(x)
        ln.set_ydata(y)
        # re-render the artist, updating the canvas state, but not the screen
        ax.draw_artist(ln)
        # copy the image to the GUI state, but screen might not be changed yet
        fig.canvas.blit(fig.bbox)
        # flush any pending GUI events, re-painting the screen if needed
        fig.canvas.flush_events()
        # you can put a pause in if you want to slow things down
        plt.pause(.03)

    plt.show()


def plot_velocity():
    """
        Plots the graphs of each particle's velocity per test run

        :return: None
    """

    x = [k for k in range(1000)]
    y = {}

    for particle in particle_history:
        y[particle.id] = []

    for n in range(tests):
        for particle in particle_history:
            y[particle.id].append((math.sqrt(particle_history[particle][n][2]**2+particle_history[particle][n][3]**2)))

    plt.cla()
    for key in y:
        plt.plot(x, y[key])
    plt.show()


def plot_positions():

    x = [k for k in range(1000)]
    y = {}

    for particle in particle_history:
        y[particle.id] = []

    for n in range(tests):
        for particle in particle_history:
            y[particle.id].append(math.dist((0, 0), particle_history[particle][n][0]))

    plt.cla()
    for key in y:
        plt.plot(x, y[key])
    plt.show()


# Number of tests
tests = 1000

# Range of function
x_range = (-5, 5)
y_range = (-2, 5)

# Velocity cap
v_max = 4 / 50

# Particle Swarm Optimisation (PSO) constants
b_pso, c_pso = 2, 2
a_pso = 0.9

# A constant subtracted from a_pso after each test that is run (0.9 - 0.4) range
d = 0.5 / tests

particle_history = {}
parameter_history = {}

particles = [Particle(x, -5, 5) for x in range(20)]

if __name__ == '__main__':

    cost_function = cost_rastrigin

    # Initialise an empty list to contain each particle's history
    for particle in particles:
        particle_history[particle] = []

    # Initialise all personal best positions to their initial position
    for particle in particles:
        particle.sp_best = particle.pos
        particle_history[particle].append((particle.pos[0], particle.pos[1], particle.v[0], particle.v[1]))

    for i in range(tests):
        print("Iteration: " + str(i))
        update_gb(particles, cost_function)

        # Save history of parameters
        parameter_history[i] = (a_pso, b_pso, c_pso)

        for particle in particles:

            cost = cost_function(particle.pos[0], particle.pos[1])

            if cost < particle.lowest_cost:
                particle.lowest_cost = cost
                particle.sp_best = particle.pos

            x_update, y_update = s_p(particle)
            v_x_update, v_y_update = v_p(particle)

            particle.f = cost
            particle.pos = (x_update, y_update)
            particle.v = (v_x_update, v_y_update)
            particle_history[particle].append((particle.pos[0], particle.pos[1], particle.v[0], particle.v[1]))

        a_pso -= d

    # Final positions and costs
    for particle in particles:
        print(particle.pos)
        print(particle.f)
        print()

    plot_graphs(cost_function)
    # print(particles[0].gb_best)
    # plot_velocity()
    # plot_positions()
