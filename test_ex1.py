from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg
import time
import logging

LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

logging.basicConfig(filename='my.log',
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)


logging.info("Go Go Go!")
refine_num = 32
LSolver = cg

coarse_mesh = Mesh(16)
fine_mesh = coarse_mesh.get_refined_mesh(refine_num)

co_on_fine_mesh = Variable(fine_mesh, Variable.TYPE_DIC["zero-order"])
co_on_fine_mesh.data = np.ones(co_on_fine_mesh.data.shape)
co_on_coarse_mesh = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
co_on_coarse_mesh.data = np.ones(co_on_coarse_mesh.data.shape)

start = time.time()
stiff_mat_ff = get_stiffness_matrix_opt(co_on_fine_mesh)
stiff_mat_cc = get_stiffness_matrix_opt(co_on_coarse_mesh)

mass_mat_f2f2 = get_mass_matrix_opt(fine_mesh,
                                    Variable.TYPE_DIC["first-order-zeroBC"],
                                    Variable.TYPE_DIC["first-order-zeroBC"])

mass_mat_f2f0 = get_mass_matrix_opt(fine_mesh,
                                    Variable.TYPE_DIC["first-order-zeroBC"],
                                    Variable.TYPE_DIC["zero-order"])

mass_mat_c2c2 = get_mass_matrix_opt(coarse_mesh,
                                    Variable.TYPE_DIC["first-order-zeroBC"],
                                    Variable.TYPE_DIC["first-order-zeroBC"])

mass_mat_c2c0 = get_mass_matrix_opt(coarse_mesh,
                                    Variable.TYPE_DIC["first-order-zeroBC"],
                                    Variable.TYPE_DIC["zero-order"])
end = time.time()
logging.info(
    "Finishing constructing Mats needed, time consuming=%.3fs.", end-start)

mass_mat_c2f2 = np.zeros(
    (coarse_mesh.inner_node_count, fine_mesh.inner_node_count))
start = time.time()
for i in range(coarse_mesh.inner_node_count):
    _base_c2 = Variable(coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"])
    _base_c2.data[i] = 1.0
    _base_c2_refined = _base_c2.project_to_refined_mesh(refine_num)
    mass_mat_c2f2[i, :] = mass_mat_f2f2.dot(_base_c2_refined.data)
A = np.zeros(mass_mat_c2f2.shape)
for j in range(fine_mesh.inner_node_count):
    A[:, j], info = LSolver(mass_mat_c2c2, mass_mat_c2f2[:, j])
end = time.time()
logging.info("Finishing constructing Mat A, time consuming=%.3fs.", end-start)

start = time.time()
B = np.zeros((fine_mesh.inner_node_count, coarse_mesh.elem_count))
for j in range(coarse_mesh.elem_count):
    _base_c0 = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
    _base_c0.data[j] = 1.0
    _base_c0_refined = _base_c0.project_to_refined_mesh(refine_num)
    B[:, j] = mass_mat_f2f0.dot(_base_c0_refined.data)
    B[:, j], info = LSolver(stiff_mat_ff, B[:, j])
end = time.time()
logging.info("Finishing constructing Mat B, time consuming=%.3fs.", end-start)

diff = stiff_mat_cc.toarray().dot(A.dot(B)) - mass_mat_c2c0
logging.info("The difference between in 2-norm is%.5f.",
             np.linalg.norm(diff, ord=2))
