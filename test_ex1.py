from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg, gmres, spsolve, LinearOperator, spilu
from scipy.sparse import csr_matrix
import time
import datetime
import logging
import sys


def _get_preconditioner(op: csr_matrix):
    M = spilu(op.tocsc())
    def Mx(x): return M.solve(x)
    return LinearOperator(op.shape, Mx)


base_grid = 2
refine_num = 2
LSolver = cg

try:
    if (len(sys.argv) >= 2):
        base_grid = int(sys.argv[1])
        assert base_grid >= 2
    if (len(sys.argv) >= 3):
        refine_num = int(sys.argv[2])
        assert refine_num >= 2
except:
    print("Invaild arguments, use default values instead.")
    base_grid = 16
    refine_num = 32

cfg = "b{0:d}r{1:d}{2:s}".format(base_grid, refine_num, LSolver.__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

NOW = datetime.datetime.now()
logging.basicConfig(filename="log/"+cfg+NOW.strftime("-%m-%d-%Y-%H-%M-%S")+'.log',
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)
logging.info("Go Go Go! with config base_grid=%d, refine_num=%d, solver=%s.",
             base_grid, refine_num, LSolver.__name__)

coarse_mesh = Mesh(base_grid)
fine_mesh = coarse_mesh.get_refined_mesh(refine_num)

co_on_fine_mesh = Variable(fine_mesh, Variable.TYPE_DIC["zero-order"])
co_on_fine_mesh.data = np.ones(co_on_fine_mesh.data.shape)
co_on_coarse_mesh = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
co_on_coarse_mesh.data = np.ones(co_on_coarse_mesh.data.shape)

start = time.time()
stiff_mat_ff = get_stiffness_matrix_opt(co_on_fine_mesh)
stiff_mat_cc = get_stiffness_matrix_opt(co_on_coarse_mesh)
pre_ff = _get_preconditioner(stiff_mat_ff)
pre_cc = _get_preconditioner(stiff_mat_cc)

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
    "Finishing constructing Mats needed, consuming time=%.3fs.", end-start)

"""
mass_mat_c2f2 = np.zeros(
    (coarse_mesh.inner_node_count, fine_mesh.inner_node_count))
start = time.time()
for i in range(coarse_mesh.inner_node_count):
    _base_c2 = Variable(coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"])
    _base_c2.data[i] = 1.0
    _base_c2_refined = _base_c2.project_to_refined_mesh(refine_num)
    mass_mat_c2f2[i, :] = mass_mat_f2f2.dot(_base_c2_refined.data)
with open("data/"+cfg+"-mat-c2f2.npy", "w") as f:
    np.save(f.name, arr=mass_mat_c2f2)
A = np.zeros(mass_mat_c2f2.shape)
for j in range(fine_mesh.inner_node_count):
    # A[:, j], info = LSolver(mass_mat_c2c2, mass_mat_c2f2[:, j])
    A[:, j], info = LSolver(mass_mat_c2c2, mass_mat_c2f2[:, j])
    if (info > 0):
        logging.critical("Solver fails, iteration num=%d.", info)
        raise RuntimeError
    elif (info < 0):
        logging.critical("Invaild input.")
        raise ValueError
end = time.time()
logging.info("Finishing constructing Mat A, consuming time=%.3fs.", end-start)
"""
mass_mat_c2f2_R = np.zeros(
    (coarse_mesh.inner_node_count, fine_mesh.inner_node_count))
start = time.time()
H = 1.0 / float(coarse_mesh.M)
for i in range(coarse_mesh.inner_node_count):
    _base_c2 = Variable(coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"])
    _base_c2.data[i] = 1.0 / H / H
    _base_c2_refined = _base_c2.project_to_refined_mesh(refine_num)
    mass_mat_c2f2_R[i, :] = mass_mat_f2f2.dot(_base_c2_refined.data)
with open("data/"+cfg+"-mat-c2f2-R.npy", "w") as f:
    np.save(f.name, arr=mass_mat_c2f2_R)
A = np.zeros(mass_mat_c2f2_R.shape)
op = mass_mat_c2c2 / (H * H)
for j in range(fine_mesh.inner_node_count):
    # A[:, j], info = LSolver(mass_mat_c2c2, mass_mat_c2f2[:, j])
    A[:, j], info = LSolver(op, mass_mat_c2f2_R[:, j])
    if (info > 0):
        logging.critical("Solver fails, iteration num=%d.", info)
        raise RuntimeError
    elif (info < 0):
        logging.critical("Invaild input.")
        raise ValueError
end = time.time()
logging.info("Finishing constructing Mat A, consuming time=%.3fs.", end-start)

start = time.time()
B = np.zeros((fine_mesh.inner_node_count, coarse_mesh.elem_count))
batch_start = time.time()
solver_consuming_time = 0.0
for j in range(coarse_mesh.elem_count):
    _base_c0 = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
    _base_c0.data[j] = 1.0
    _base_c0_refined = _base_c0.project_to_refined_mesh(refine_num)
    B[:, j] = mass_mat_f2f0.dot(_base_c0_refined.data)
    solver_start = time.time()
    B[:, j], info = LSolver(stiff_mat_ff, B[:, j])
    solver_end = time.time()
    solver_consuming_time += (solver_end - solver_start)
    if (info > 0):
        logging.critical("Solver fails, iteration num=%d.", info)
        raise RuntimeError
    elif (info < 0):
        logging.critical("Invaild input.")
        raise ValueError
    if (j % coarse_mesh.M == coarse_mesh.M-1):
        batch_end = time.time()
        logging.info("\tBatch %d/%d completed, consuming time=%.3fs, solver consuming time=%.3fs;",
                     j//coarse_mesh.M+1, coarse_mesh.M, batch_end-batch_start, solver_consuming_time)
        batch_start = time.time()
        solver_consuming_time = 0.0

end = time.time()
logging.info("Finishing constructing Mat B, consuming time=%.3fs.", end-start)
C = np.zeros((coarse_mesh.inner_node_count, coarse_mesh.elem_count))
for j in range(coarse_mesh.elem_count):
    _base_c0 = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
    _base_c0.data[j] = 1.0
    C[:, j] = mass_mat_c2c0.dot(_base_c0.data)
    C[:, j], info = LSolver(stiff_mat_cc, C[:, j])


mat_AB = np.matmul(A, B)
diff = np.matmul(stiff_mat_cc.toarray(), mat_AB) - mass_mat_c2c0.toarray()
with open("data/"+cfg+"-mat-AB.npy", "w") as f:
    np.save(f.name, arr=mat_AB)
with open("data/"+cfg+"-mat-C.npy", "w") as f:
    np.save(f.name, arr=C)
with open("data/"+cfg+"-mat-A.npy", "w") as f:
    np.save(f.name, arr=A)
with open("data/"+cfg+"-mat-B.npy", "w") as f:
    np.save(f.name, arr=B)
logging.info("The absolute difference in 2-norm=%.5f.",
             np.linalg.norm(diff, ord=2))
logging.info("The U_H difference in 2-norm=%.5f.",
             np.linalg.norm(mat_AB-C, ord=2))
logging.info("The relative difference in 2-norm=%.5f.",
             np.linalg.norm(diff, ord=2)/np.linalg.norm(mass_mat_c2c0.toarray(), ord=2))
