import numpy as np
import matplotlib.pyplot as plt
from StructuredMesh2D import Mesh, Variable, get_stiffness_matrix_opt, get_mass_matrix_opt
from matplotlib.ticker import MaxNLocator

x = np.linspace(0.9, 3.0, num=211)
y = np.linspace(-2.1, 2.1, num=421)
XX, YY = np.meshgrid(x, y)
data = np.load("data/error.npy")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal', 'box')
_max, _min = np.max(data), np.min(data)
levels = MaxNLocator(nbins=15).tick_values(-1.0e-5, _max)
cf = ax.contourf(XX, YY, data, levels=levels,
                 cmap="viridis", vmin=-1.0e-5, vmax=_max)
fig.colorbar(cf, ax=ax)
plt.show()