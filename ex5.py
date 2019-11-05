from StructuredMesh2D import Mesh, Variable, get_stiffness_matrix_opt, get_mass_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg
import logging
import time

LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(filename='my.log',
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)

"""
 -Laplace u = f in Omega
 u= 0 on partial Omega

 u = x(1-x)y(1-y)
 f = 2y(1-y)+2x(1-x)

 u = sin(pi x)sin(pi y)
 f = 2.0 pi pi sin(pi x)sin(pi y) 
"""

Pi = np.pi
Sin = np.sin
Cos = np.cos


def f_func(x: float, y: float) -> float:
    return -4*Pi**2*Cos(2*Pi*x)*Cos(2*Pi*y)*Sin(2*Pi*y) + \
        4*Pi**2*Cos(2*Pi*x)*Sin(2*Pi*x)*Sin(2*Pi*y) + \
        4*Pi**2*(6 + Cos(2*Pi*x))*Sin(2*Pi*x)*Sin(2*Pi*y) + \
        4*Pi**2*Sin(2*Pi*x)*(6 + Sin(2*Pi*x))*Sin(2*Pi*y) - \
        8*Pi**2*Cos(2*Pi*x)*Cos(2*Pi*y)*(1 + Sin(2*Pi*y))


def u_func(x: float, y: float) -> float:
    # return (x-0.2)**2 + (y-0.8)**2
    # return 2.0 * np.pi * np.pi * np.sin(np.pi*x) * np.sin(np.pi*y)
    return Sin(2*Pi*x) * Sin(2*Pi*y)


def A0_func(x: float, y: float) -> float:
    return 6. + Cos(2*Pi*x)


def A1_func(x: float, y: float) -> float:
    return 6. + Sin(2*Pi*x)


def A2_func(x: float, y: float) -> float:
    return 1. + Sin(2*Pi*y)


"""
M16 = Mesh(1024)
co_ones = Variable(M16, Variable.TYPE_DIC["zero-order"])
co_ones.data = np.ones(co_ones.data.shape)
d_laplace = get_stiffness_matrix_opt(co_ones)
mass_mat = get_mass_matrix_opt(
    M16, Variable.TYPE_DIC["first-order-zeroBC"], Variable.TYPE_DIC["zero-order"])
var_f = Variable(M16, Variable.TYPE_DIC["zero-order"])
var_f.evaluation_data_by_func(rhs_f)
rhs = mass_mat.dot(var_f.data)
var_u = Variable(M16, Variable.TYPE_DIC["first-order-zeroBC"])
var_u.evaluation_data_by_func(u_f)
logging.info("Begin to solve linear system.")
start = time.time()
u_cg, info = cg(d_laplace, rhs)
end = time.time()
logging.info("Finish solving linear system, consuming time=%.3fs.", end-start)
logging.debug("error in inf-norm=%.5f.",
              np.linalg.norm(u_cg-var_u.data, ord=np.inf))
"""
mesh = Mesh(32)
u_var = Variable(mesh, Variable.TYPE_DIC["first-order-zeroBC"])
u_var.evaluation_data_by_func(u_func)

f_var = Variable(mesh, Variable.TYPE_DIC["zero-order"])
f_var.evaluation_data_by_func(f_func)

A_var = Variable(mesh, Variable.TYPE_DIC["zero-order-matrix"])
A_var.evaluation_data_by_func(
    None, func0=A0_func, func1=A1_func, func2=A2_func)

stiffness_mat = get_stiffness_matrix_opt(A_var)
mass_mat = get_mass_matrix_opt(mesh, Variable.TYPE_DIC["first-order-zeroBC"],
                               Variable.TYPE_DIC["zero-order"])

rhs = mass_mat.dot(f_var.data)
logging.info("Begin to solve linear system.")
start = time.time()
u_cg, info = cg(stiffness_mat, rhs)
u_comp_var = u_var.init_by_copy()
u_comp_var.data = u_cg
end = time.time()
logging.info("Finish solving linear system, consuming time=%.3fs.", end-start)
error = u_comp_var - u_var
error.get_plot()
logging.info("error in inf-norm=%.5f.",
             np.linalg.norm(error.data, ord=np.inf) / np.linalg.norm(u_var.data, ord=np.inf))
