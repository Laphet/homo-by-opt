from EllipticCoeffOpt import EllipticCoeffOpt as OPT
from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
import sys
import time
import datetime
import random
import logging
from scipy.sparse.linalg import cg

SPLSolver = cg
PI = np.pi
ZERO_ORDER_MATRIX = Variable.TYPE_DIC["zero-order-matrix"]
FIRST_ORDER_ZEROBC = Variable.TYPE_DIC["first-order-zeroBC"]
ZERO_ORDER = Variable.TYPE_DIC["zero-order"]


def cell_func(x: float, y: float):
    return 6.0 + np.cos(2.*PI*x) + np.sin(2.*PI*x) + np.cos(2.*PI*y) + np.sin(2.*PI*y)


def periodic_func(x: float, y: float, cell_num: int):
    x_ = (cell_num*x) % 1.0
    y_ = (cell_num*y) % 1.0
    return cell_func(x_, y_)


def zero_func(x: float, y: float, args=None):
    return 0.0

def random_func(x: float, y: float, U: float):
    return random.uniform(0., U)

cell_num = 4
coarse_elem_num = 4
refine_elem_num = 16
data_num = 5

try:
    if (len(sys.argv) >= 2):
        cell_num = int(sys.argv[1])
        assert cell_num >= 2
    if (len(sys.argv) >= 3):
        coarse_elem_num = int(sys.argv[2])
        assert coarse_elem_num >= 2
    if (len(sys.argv) >= 4):
        refine_elem_num = int(sys.argv[3])
        assert (refine_elem_num >= 4 and refine_elem_num %
                coarse_elem_num == 0)
    if (len(sys.argv) >= 5):
        data_num = int(sys.argv[4])
        assert data_num >= 4
except:
    print("Invaild arguments, use default values instead.")
    cell_num = 64
    coarse_elem_num = 8
    refine_elem_num = 64 * 32
    data_num = 5

refine_num = refine_elem_num // coarse_elem_num

cfg = "Ce{:d}Bg{:d}Rf{:d}Dn{:d}-{:s}-".format(cell_num,
                                              coarse_elem_num,
                                              refine_num,
                                              data_num,
                                              datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S"))

LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
logging.basicConfig(filename="log/{:s}.log".format(cfg),
                    level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt=DATE_FORMAT)
logging.info("Go Go Go!")

coarse_mesh = Mesh(coarse_elem_num)
refine_mesh = Mesh(refine_elem_num)

refine_co = Variable(refine_mesh, ZERO_ORDER_MATRIX)
refine_co.evaluation_data_by_func(None, func0=periodic_func,
                                  func1=periodic_func,
                                  func2=zero_func,
                                  args=cell_num)

refine_stiff_mat = get_stiffness_matrix_opt(refine_co)
refine_mass_mat_rhs = get_mass_matrix_opt(
    refine_mesh, FIRST_ORDER_ZEROBC, ZERO_ORDER)
random.seed()
opt = OPT(coarse_mesh, refine_mesh, 6.0 - 2.0*np.sqrt(2.0))
for id in range(data_num):
    source_term = Variable(coarse_mesh, ZERO_ORDER)
    source_term.evaluation_data_by_func(random_func, args=float(coarse_mesh.M)**2)
    refine_source_term = source_term.project_to_refined_mesh(refine_num)
    rhs = refine_mass_mat_rhs.dot(refine_source_term.data)
    refine_solution, info = SPLSolver(refine_stiff_mat, rhs)
    opt.add_homo_solution_ref(refine_solution, source_term)
A_init = refine_co.average_to_coarsen_mesh(coarse_mesh)
A_homo, val = opt.opt_routine(A_init)
np.save("data/{:s}.npy".format(cfg), A_homo.data)
