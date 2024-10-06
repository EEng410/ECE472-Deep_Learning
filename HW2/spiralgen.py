import numpy as np
import matplotlib.pyplot as plt
import sys
rng = np.random.default_rng()
theta = rng.uniform(0, 6*np.pi, 500)
r = 3 * theta
var = 1
noise = rng.normal(0, var, [4, 500])

x0 = r * np.cos(theta)
y0 = r * np.sin(theta)

x1 = -r * np.cos(theta)
y1 = -r * np.sin(theta)

coords = np.array([x0, y0, x1, y1])
coords = coords + noise

plt.scatter(coords[0, :], coords[1, :])
plt.scatter(coords[2, :], coords[3, :])
plt.show()

#Two  lines to make our compiler able to draw:
plt.savefig(sys.stdout.buffer)
sys.stdout.flush()
