# rastrigin_graph.py

import numpy as np
import matplotlib.pyplot as plt



n = 100
X = np.linspace(-5, 5, n)     
Y = np.linspace(-5, 5, n)     
X, Y = np.meshgrid(X, Y) 

Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
  (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
 
fig = plt.figure() 
pcm = plt.contourf(X,Y,Z, levels=50, cmap="jet")
fig.colorbar(pcm)
plt.show()