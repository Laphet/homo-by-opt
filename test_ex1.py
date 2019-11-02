from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg, gmres, spsolve, LinearOperator
from scipy.sparse import csr_matrix
import time
import datetime
import logging
import sys


def _get_preconditioner(op: csr_matrix):
    assert (op.shape[0] == op.shape[1])
    N = op.shape[0]
    diagonal_mat = csr_matrix(
        (op.diagonal(), (np.arange(N), np.arange(N))), shape=(N, N))

    def M_x(x): return spsolve(diagonal_mat, x)
    M = LinearOperator((N, N), M_x)
    return M


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

cfg = "b{0:d}r{1:d}".format(base_grid, refine_num)
LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"

NOW = datetime.datetime.now()
logging.basicConfig(filename="log/"+cfg+NOW.strftime("-%m-%d-%Y-%H-%M-%S")+'.log',
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)
logging.info("Go Go Go! with config base_grid=%d, refine_num=%d.",
             base_grid, refine_num)

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
    A[:, j], info = LSolver(mass_mat_c2c2, mass_mat_c2f2[:, j], M=pre_cc)
end = time.time()
logging.info("Finishing constructing Mat A, consuming time=%.3fs.", end-start)

start = time.time()
B = np.zeros((fine_mesh.inner_node_count, coarse_mesh.elem_count))
batch_start = time.time()
for j in range(coarse_mesh.elem_count):
    _base_c0 = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
    _base_c0.data[j] = 1.0
    _base_c0_refined = _base_c0.project_to_refined_mesh(refine_num)
    B[:, j] = mass_mat_f2f0.dot(_base_c0_refined.data)
    B[:, j], info = LSolver(stiff_mat_ff, B[:, j], M=pre_ff)
    if (j % coarse_mesh.M == coarse_mesh.M-1):
        batch_end = time.time()
        logging.info("\tBatch %d/%d completed, consuming time=%.3fs;",
                     j//coarse_mesh.M+1, coarse_mesh.M, batch_end-batch_start)
        batch_start = time.time()

end = time.time()
logging.info("Finishing constructing Mat B, consuming time=%.3fs.", end-start)
mat_AB = np.matmul(A, B)
diff = np.matmul(stiff_mat_cc.toarray(), mat_AB) - mass_mat_c2c0.toarray()
with open("data/"+cfg+"-mat-AB.npy", "w") as f:
    np.save(f.name, arr=mat_AB)
logging.info("The absolute difference in 2-norm=%.5f.",
             np.linalg.norm(diff, ord=2))
logging.info("The relative difference in 2-norm=%.5f.",
             np.linalg.norm(diff, ord=2)/np.linalg.norm(mass_mat_c2c0.toarray(), ord=2))
