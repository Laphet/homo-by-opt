from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg, gmres
from scipy.linalg import inv
import logging
import time
import datetime

SPLSolver = cg
DLSolver = np.linalg.solve
INV = inv
PI = np.pi


class EllipticCoeffOpt:
    EPSILON = 1.0e-5
    MAX_LINE_SEARCH = 10
    MAX_OPT_STEPS = 1024

    def __init__(self, _coarse_mesh: Mesh, _refine_co: Variable):
        _refine_mesh = _refine_co.mesh
        assert (_refine_mesh.M % _coarse_mesh.M == 0)
        refine_num = _refine_mesh.M // _coarse_mesh.M
        self.coarse_mesh = _coarse_mesh
        self.refine_mesh = _refine_mesh
        self.refine_co = Variable(
            _refine_mesh, Variable.TYPE_DIC["zero-order-matrix"])
        if (_refine_co.type == Variable.TYPE_DIC["zero-order"]):
            self.refine_co.data[0, :] = _refine_co.data[:]
            self.refine_co.data[1, :] = _refine_co.data[:]
        elif (_refine_co.type == Variable.TYPE_DIC["zero-order-matrix"]):
            self.refine_co.data = _refine_co.data
        self.min_lambda = self.refine_co.get_min_eigen()
        assert (self.min_lambda >
                EllipticCoeffOpt.EPSILON), "Too small eigens for coefficients appear!"
        self.homo_solution_ref_count = 0
        self.rhs_data = np.zeros((1, _coarse_mesh.inner_node_count))
        self.homo_solution_ref_data = np.zeros(
            (1, _coarse_mesh.inner_node_count))
        self.obj_weight = []
        self.co2mass_trans_tensor = _coarse_mesh.get_stiffness_trans_tensor()

        self.coarse_mass_mat = get_mass_matrix_opt(
            _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"], Variable.TYPE_DIC["first-order-zeroBC"])
        self.coarse_mass_mat_rhs = get_mass_matrix_opt(
            _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"], Variable.TYPE_DIC["zero-order"])
        self.refine2coarse_trans_mass_mat_scaled = np.zeros(
            (_coarse_mesh.inner_node_count, _refine_mesh.inner_node_count))
        refine_mass_mat = get_mass_matrix_opt(
            _refine_mesh, Variable.TYPE_DIC["first-order-zeroBC"], Variable.TYPE_DIC["first-order-zeroBC"])
        H = 1.0 / float(_coarse_mesh.M)
        for i in range(_coarse_mesh.inner_node_count):
            coarse_lagrange_base = Variable(
                _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"])
            coarse_lagrange_base.data[i] = 1.0
            coarse2refine_lagrange_base = coarse_lagrange_base.project_to_refined_mesh(
                refine_num)
            self.refine2coarse_trans_mass_mat_scaled[i, :] = 1.0 / H / H * \
                refine_mass_mat.dot(coarse2refine_lagrange_base.data)

        self.__inv_stiff_mat_A: np.ndarray = None
        self.__GA_data: np.ndarray = None

    def add_homo_solution_ref(self, refine_solution: np.ndarray, source_term: Variable):
        assert (refine_solution.shape[0] == self.refine_mesh.inner_node_count)
        new_rhs = self.coarse_mass_mat_rhs.dot(source_term.data)
        H = 1.0 / float(self.coarse_mesh.M)
        new_homo_solution_ref, info = SPLSolver(
            1.0 / H / H * self.coarse_mass_mat, self.refine2coarse_trans_mass_mat_scaled.dot(refine_solution))
        assert (info == 0), "The solver fails!"
        if (self.homo_solution_ref_count == 0):
            self.rhs_data[0, :] = new_rhs
            self.homo_solution_ref_data[0, :] = new_homo_solution_ref
        else:
            np.vstack((self.rhs_data, new_rhs))
            np.vstack((self.homo_solution_ref_data, new_homo_solution_ref))
        self.homo_solution_ref_count += 1
        weight = self.coarse_mass_mat.dot(
            new_homo_solution_ref) @ new_homo_solution_ref
        self.obj_weight.append(weight)
        logging.info(
            "Add a homogenized solution reference.")

    def get_co_var_from_vec(self, vec: np.ndarray):
        co = Variable(self.coarse_mesh, Variable.TYPE_DIC["zero-order-matrix"])
        co.data = vec.reshape((3, -1), order='F')
        return co

    def _get_subroutine_quad_opt(self, A: Variable):
        G = np.zeros((Variable._MATRIX_ENTRY_PER_ELEM_*self.coarse_mesh.elem_count,
                      Variable._MATRIX_ENTRY_PER_ELEM_*self.coarse_mesh.elem_count))
        b = np.zeros((Variable._MATRIX_ENTRY_PER_ELEM_ *
                      self.coarse_mesh.elem_count, ))
        for homo_ind in range(self.homo_solution_ref_count):
            diff = self.homo_solution_ref_data[homo_ind, :] - \
                self.__GA_data[homo_ind, :]
            dG_AB = self.__inv_stiff_mat_A @ \
                np.tensordot(self.co2mass_trans_tensor, self.__GA_data[homo_ind, :],
                             axes=([1, ], [0, ]))
            b += (diff @ (self.coarse_mass_mat @ dG_AB)) / \
                self.obj_weight[homo_ind]
            G += (dG_AB.transpose() @ self.coarse_mass_mat @ dG_AB) / \
                self.obj_weight[homo_ind]
        G = G / float(self.homo_solution_ref_count)
        b = b / float(self.homo_solution_ref_count)
        try:
            co_vec = DLSolver(G, b)
        except np.linalg.LinAlgError:
            norm2_dG_AB = np.linalg.norm(dG_AB, ord=2)
            norminf_dF_AB = np.linalg.norm(b, ord=np.inf)
            logging.critical(
                "\tSub optimization routine has evolved to singular, " +
                "singular operator norm=%.5f, " +
                "current grad of obj function=%.5f," +
                "a zero direction returned.",
                norm2_dG_AB, norminf_dF_AB)
            co_vec = np.zeros(b.shape)
        finally:
            return self.get_co_var_from_vec(co_vec)

    def project_to_feasible_set(self, A: Variable):
        count = 0
        for elem_ind in range(self.coarse_mesh.elem_count):
            l = Variable._get_eigens_(A.data[:, elem_ind])
            if (np.min(l) < self.min_lambda):
                count += 1
                A.data[:-1, elem_ind] += self.min_lambda - np.min(l)

        if (count > 0):
            logging.info(
                "\tProjection performed, %d elems have been shifted.", count)

    def obj_func_val(self, A: Variable):
        val = 0.0
        stiff_A = get_stiffness_matrix_opt(A)
        start = time.time()
        self.__inv_stiff_mat_A: np.ndarray = INV(stiff_A.toarray())
        end = time.time()
        logging.info(
            "\tPerform a matrix inverse, consuming time=%.3fs.", end-start)
        self.__GA_data = np.zeros(
            (self.homo_solution_ref_count, self.coarse_mesh.inner_node_count))
        for homo_ind in range(self.homo_solution_ref_count):
            u_coarse = self.__inv_stiff_mat_A.dot(self.rhs_data[homo_ind, :])
            self.__GA_data[homo_ind, :] = u_coarse
            diff = self.homo_solution_ref_data[homo_ind, :]-u_coarse
            a = self.coarse_mass_mat.dot(diff) @ diff
            val += a / self.obj_weight[homo_ind]
        return 0.5 / float(self.homo_solution_ref_count) * val

    def opt_routine(self):
        A_curr = self.refine_co.average_to_coarsen_mesh(self.coarse_mesh)
        obj_val_curr = self.obj_func_val(A_curr)
        for opt_step_ind in range(EllipticCoeffOpt.MAX_OPT_STEPS):
            start = time.time()
            B = self._get_subroutine_quad_opt(A_curr)
            for line_search_ind in range(EllipticCoeffOpt.MAX_LINE_SEARCH):
                A_next = A_curr.direct_add(B)
                self.project_to_feasible_set(A_next)
                obj_val_next = self.obj_func_val(A_next)
                if (-EllipticCoeffOpt.EPSILON**2 > obj_val_curr - obj_val_next > EllipticCoeffOpt.EPSILON):
                    break
                else:
                    B.direct_div(2.0)

            if (line_search_ind == EllipticCoeffOpt.MAX_LINE_SEARCH):
                logging.critical("Line search reaches max iteration, careful!")
                A_next = A_curr
                obj_val_next = obj_val_curr

            descent = A_next.diff_inf_norm(A_curr)
            if (descent < EllipticCoeffOpt.EPSILON):
                logging.info("Optimization process ends successfully.")
                return A_next, obj_val_next
            else:
                A_curr = A_next
                obj_val_curr = obj_val_next
                end = time.time()
                logging.info("Optimization step index=%d, " +
                             "coefficient descends %.5f, " +
                             "opt cycle consuming time=%.3f.",
                             opt_step_ind, descent, end-start)


if __name__ == "__main__":
    NOW = datetime.datetime.now()
    LOG_FORMAT = "%(asctime)s %(levelname)s \t%(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S"
    logging.basicConfig(filename="log/"+NOW.strftime("-%m-%d-%Y-%H-%M-%S")+'.log',
                        level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt=DATE_FORMAT)
    logging.info("Go Go Go!")
    coarse_mesh = Mesh(16)
    refine_mesh = coarse_mesh.get_refined_mesh(32)
    import random
    def func0(x, y, args=None): return 4.0+np.cos(x*16.0*PI)+np.sin(y*16.0*PI)
    refine_co = Variable(refine_mesh, Variable.TYPE_DIC["zero-order"])
    refine_co.evaluation_data_by_func(func0)
    refine_co.get_plot("fig/refine_co.png")
    opt = EllipticCoeffOpt(coarse_mesh, refine_co)
    refine_stiff_mat = get_stiffness_matrix_opt(refine_co)
    refine_rhs_mass_mat = get_mass_matrix_opt(
        refine_co, Variable.TYPE_DIC["first-order"], Variable.TYPE_DIC["zero-order"])
    data_num = 4
    source_term_list = random.sample(list(coarse_mesh.elem_count), data_num)
    H = 1.0 / float(coarse_mesh.M)
    for id in range(data_num):
        source_term = Variable(coarse_mesh, Variable.TYPE_DIC["zero-order"])
        source_term.data[id] = 1.0 / H / H
        refine_source_term = source_term.project_to_refined_mesh(32)
        rhs = refine_rhs_mass_mat.dot(refine_source_term.data)
        refine_solution, info = SPLSolver(refine_stiff_mat, rhs)
        opt.add_homo_solution_ref(refine_solution, source_term)
    A_homo, obj_val = opt.opt_routine()
    co_homo = opt.get_co_var_from_vec(A_homo)
    A_homo.get_plot("fig/homo_co_0.png", component=0)
