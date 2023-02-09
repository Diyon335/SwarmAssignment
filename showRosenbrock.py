import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def z_rosenbrock(x, y):
    """
    The Rosenbrock cost function
    :param particle: Particle object
    :return: Returns an integer indicating the cost
    """

    a = 0
    b = 100

    return (a - x) ** 2 + (b * (y - x ** 2) ** 2)

n = 100
X = np.linspace(-2, 2, n)     
Y = np.linspace(-1, 3, n)
X, Y = np.meshgrid(X, Y)

Z = z_rosenbrock(X,Y)

fig = plt.figure() 

pcm = plt.pcolor(X, Y, Z,
                   norm=colors.LogNorm(vmin=Z.min(), vmax=Z.max()),
                   cmap='jet', shading='auto')
fig.colorbar(pcm, extend='max')
plt.show()

