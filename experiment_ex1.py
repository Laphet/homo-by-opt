import numpy as np
import matplotlib.pyplot as plt
from StructuredMesh2D import Mesh, Variable, get_stiffness_matrix_opt, get_mass_matrix_opt
from matplotlib.ticker import MaxNLocator


def _co(a: float, b: float):
    def func0(x, y): return a
    def func1(x, y): return b
    return func0, func1


base_grid = 16
coarse_mesh = Mesh(base_grid)
mat_AB = np.load("data/b16r32cg-mat-C.npy")
mass_mat_c2c0 = get_mass_matrix_opt(coarse_mesh,
                                    Variable.TYPE_DIC["first-order-zeroBC"],
                                    Variable.TYPE_DIC["zero-order"]).toarray()

x = np.linspace(0.9, 3.0, num=211)
y = np.linspace(-2.1, 2.1, num=421)
XX, YY = np.meshgrid(x, y)
data = np.zeros(XX.shape)
stiff_diff = np.zeros(XX.shape)

co = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
co.data = np.ones(co.data.shape)
stiff_real = get_stiffness_matrix_opt(co)

for i in range(XX.shape[0]):
    for j in range(XX.shape[1]):
        if (XX[i, j]+YY[i, j] >= 0.9 and XX[i, j]-YY[i, j] >= 0.9):
            co = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order-matrix"])
            func0, func1 = _co(XX[i, j], YY[i, j])
            co.evaluation_data_by_func(
                None, func0=func0, func1=func0, func2=func1)
            stiff_mat_cc = get_stiffness_matrix_opt(co).toarray()
            data[i, j] = np.linalg.norm(np.matmul(
                stiff_mat_cc, mat_AB)-mass_mat_c2c0, ord=2)
            stiff_diff[i, j] = np.linalg.norm(stiff_mat_cc-stiff_real, ord=2)
        else:
            data[i, j] = -1.0
            stiff_diff[i, j] = -1.0

np.save("data/error.npy", arr=data)
np.save("data/stiff_diff.npy", arr=stiff_diff)
