import numpy as np
import matplotlib.pyplot as plt
from StructuredMesh2D import Mesh, Variable, get_stiffness_matrix_opt, get_mass_matrix_opt
from matplotlib.ticker import MaxNLocator

x = np.linspace(0.9, 3.0, num=211)
y = np.linspace(-2.1, 2.1, num=421)
XX, YY = np.meshgrid(x, y)
data = np.load("data/error.npy")
stiff_diff = np.load("data/stiff_diff.npy")

fig = plt.figure()
ax0, ax1 = fig.subplots(1, 2)
ax0.set_aspect('equal', 'box')
_max, _min = np.max(data), np.min(data)
levels = MaxNLocator(nbins=15).tick_values(-1.0e-5, _max)
cf0 = ax0.contourf(XX, YY, data, levels=levels,
                   cmap="viridis", vmin=-1.0e-5, vmax=_max)
fig.colorbar(cf0, ax=ax0)

ax1.set_aspect('equal', 'box')
__max, __min = np.max(stiff_diff), np.min(stiff_diff)
_levels = MaxNLocator(nbins=15).tick_values(-1.0e-5, __max)
cf1 = ax1.contourf(XX, YY, stiff_diff, levels=_levels,
                   cmap="viridis", vmin=-1.0e-5, vmax=__max)
fig.colorbar(cf1, ax=ax1)
plt.show()
