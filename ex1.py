import numpy as np
from StructuredMesh2D import Mesh, Variable, get_stiffness_matrix_opt, get_mass_matrix_opt
from scipy.sparse.linalg import cg


def co(x: float, y: float, args=None):
    if (np.abs(y % 0.1-0.05) < 0.01):
        return 100.0
    elif (np.abs(x % 0.1-0.05) < 0.01):
        return 100.0
    else:
        return 1.0


def f(x: float, y: float, flag: int):
    if (flag == 0):
        return 1.0
    elif (flag == 1):
        if (0.1 < x < 0.9 and 0.1 < y < 0.9):
            return 1.0 / 0.8 / 0.8
        else:
            return 0.0
    else:
        if (x < 0.5 and y < 0.5):
            return 4.0
        else:
            return 0.0


mesh = Mesh(512)

co_var = Variable(mesh, Variable.TYPE_DIC["zero-order"])
co_var.evaluation_data_by_func(co)


u_var = Variable(mesh, Variable.TYPE_DIC["first-order-zeroBC"])

stiff_mat = get_stiffness_matrix_opt(co_var)
mass_mat = get_mass_matrix_opt(
    mesh, Variable.TYPE_DIC["first-order-zeroBC"], Variable.TYPE_DIC["zero-order"])

f_var = Variable(mesh, Variable.TYPE_DIC["zero-order"])

f_var.evaluation_data_by_func(f, args=0)
rhs_vec = mass_mat.dot(f_var.data)
u_var.data, info = cg(stiff_mat, rhs_vec)
u_var.get_plot(filename="fig/u-flag-0.png")

f_var.evaluation_data_by_func(f, args=1)
rhs_vec = mass_mat.dot(f_var.data)
u_var.data, info = cg(stiff_mat, rhs_vec)
u_var.get_plot(filename="fig/u-flag-1.png")

f_var.evaluation_data_by_func(f, args=2)
rhs_vec = mass_mat.dot(f_var.data)
u_var.data, info = cg(stiff_mat, rhs_vec)
u_var.get_plot(filename="fig/u-flag-2.png")
