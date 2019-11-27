from StructuredMesh2D import Mesh, Variable, get_mass_matrix_opt, get_stiffness_matrix_opt
import numpy as np
from scipy.sparse.linalg import cg, gmres
import time
import datetime
import logging


SPLSolver = cg
DLSolver = np.linalg.solve
PI = np.pi


class EllipticCoeffOpt:
    EPSILON = 1.0e-5
    MAX_LINE_SEARCH = 18
    MAX_OPT_STEPS = 4096

    def __init__(self, _coarse_mesh: Mesh, _refine_mesh: Mesh, _min_lambda: float):
        assert (_refine_mesh.M % _coarse_mesh.M == 0)
        assert (_min_lambda > EllipticCoeffOpt.EPSILON)
        refine_num = _refine_mesh.M // _coarse_mesh.M
        # parameters
        self.coarse_mesh = _coarse_mesh
        self.refine_mesh = _refine_mesh
        self.min_lambda = _min_lambda
        H = 1.0 / float(_coarse_mesh.M)
        self.rho = H * 0.001
        # trans tensor
        self.u2coeff_trans_tensor = _coarse_mesh.get_u2coeff_trans_tensor()
        # data
        self.homo_solution_ref_count = 0
        self.rhs_data = np.zeros((1, _coarse_mesh.inner_node_count))
        self.homo_solution_ref_data = np.zeros(
            (1, _coarse_mesh.inner_node_count))
        # filter
        self.coarse_mass_mat = get_mass_matrix_opt(
            _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"],
            Variable.TYPE_DIC["first-order-zeroBC"])
        self.coarse_mass_mat_rhs = get_mass_matrix_opt(
            _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"],
            Variable.TYPE_DIC["zero-order"])
        self.refine2coarse_trans_mass_mat_scaled = np.zeros(
            (_coarse_mesh.inner_node_count, _refine_mesh.inner_node_count))
        refine_mass_mat = get_mass_matrix_opt(
            _refine_mesh, Variable.TYPE_DIC["first-order-zeroBC"],
            Variable.TYPE_DIC["first-order-zeroBC"])

        for i in range(_coarse_mesh.inner_node_count):
            coarse_lagrange_base = Variable(
                _coarse_mesh, Variable.TYPE_DIC["first-order-zeroBC"])
            coarse_lagrange_base.data[i] = 1.0
            coarse2refine_lagrange_base = coarse_lagrange_base.project_to_refined_mesh(
                refine_num)
            self.refine2coarse_trans_mass_mat_scaled[i, :] = 1.0 / H / H * \
                refine_mass_mat.dot(coarse2refine_lagrange_base.data)
        # temp data
        self.__GA_data: np.ndarray = None

    def get_u2coeff(self, u: np.ndarray):
        return np.tensordot((np.tensordot(self.u2coeff_trans_tensor, u, axes=1)), u, axes=1)

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
            self.rhs_data = np.vstack(
                (self.rhs_data, new_rhs.reshape((1, -1))))
            self.homo_solution_ref_data = np.vstack(
                (self.homo_solution_ref_data, new_homo_solution_ref.reshape((1, -1))))
        self.homo_solution_ref_count += 1
        """ V1
        weight = self.coarse_mass_mat.dot(
            new_homo_solution_ref) @ new_homo_solution_ref
        self.obj_weight.append(weight)
        """
        logging.info(
            "Add a homogenized solution reference.")

    def get_norm2_square_coeff(self, co: Variable):
        assert co.type == Variable.TYPE_DIC["zero-order-matrix"]
        norm0 = np.linalg.norm(co.data[0, :])
        norm1 = np.linalg.norm(co.data[1, :])
        norm2 = np.linalg.norm(co.data[2, :])
        H = 1.0 / float(co.mesh.M)
        return H**2 * (norm0**2 + norm1**2 + 2.0*norm2**2)

    def get_norm2_square_coeff_data(self, B: np.ndarray):
        norm0 = np.linalg.norm(B[0, :])
        norm1 = np.linalg.norm(B[1, :])
        norm2 = np.linalg.norm(B[2, :])
        H = 1.0 / float(self.coarse_mesh.M)
        return H**2 * (norm0**2 + norm1**2 + 2.0*norm2**2)

    def get_normINF_coeff(self, co: Variable):
        return np.linalg.norm(co.data.flatten(), ord=np.inf)

    def get_normINF_coeff_data(self, B: np.ndarray):
        return np.linalg.norm(B.flatten(), ord=np.inf)

    """ V2
    def get_co_var_from_vec(self, vec: np.ndarray):
        co = Variable(self.coarse_mesh, Variable.TYPE_DIC["zero-order-matrix"])
        co.data = vec.reshape((3, -1), order='F')
        return co
        

    def _get_subroutine_quad_opt(self, A: Variable):
        N_nodes = self.coarse_mesh.inner_node_count
        N0 = self.homo_solution_ref_count * N_nodes
        N1 = Variable._MATRIX_ENTRY_PER_ELEM_*self.coarse_mesh.elem_count
        G = np.zeros((N0, N1))
        b = np.zeros((N0, ))
        for homo_ind in range(self.homo_solution_ref_count):
            diff = self.homo_solution_ref_data[homo_ind,
                                               :] - self.__GA_data[homo_ind, :]
            weight = np.linalg.norm(self.homo_solution_ref_data[homo_ind, :])
            b[homo_ind * N_nodes: (homo_ind+1) * N_nodes] = diff / weight
            dG_AB = -1.0 * self.__inv_stiff_mat_A @ \
                np.tensordot(self.co2mass_trans_tensor, self.__GA_data[homo_ind, :],
                             axes=([1, ], [0, ]))
            G[homo_ind * N_nodes: (homo_ind+1) * N_nodes, :] = dG_AB / weight
        try:
            co_vec, _res, _rnk, _s = LS(G, b, cond=EllipticCoeffOpt.EPSILON)
        except:
            logging.critical(
                "\tSub LS routine fails, a zero direction returned.")
            co_vec = np.zeros(b.shape)
        finally:
            return self.get_co_var_from_vec(co_vec)
    """

    """ V1
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
    """

    def project_to_feasible_set(self, A: Variable):
        count = 0
        for elem_ind in range(self.coarse_mesh.elem_count):
            a, b, c = (A.data[0, elem_ind],
                       A.data[1, elem_ind],
                       A.data[2, elem_ind])
            s, d = a+b, a-b
            eigens = np.array([0.5*(s-np.sqrt(d**2+4.0*c**2)),
                               0.5*(s+np.sqrt(d**2+4.0*c**2))])
            if (np.min(eigens) < self.min_lambda):
                count += 1
                if (np.abs(c) < EllipticCoeffOpt.EPSILON**2):
                    a = self.min_lambda if a < self.min_lambda else a
                    b = self.min_lambda if b < self.min_lambda else b
                else:
                    l1 = self.min_lambda if eigens[0] < self.min_lambda else eigens[0]
                    l2 = self.min_lambda if eigens[1] < self.min_lambda else eigens[1]
                    v1 = np.array([d-np.sqrt(d**2+4.0*c**2), 2*c])
                    v2 = np.array([d+np.sqrt(d**2+4.0*c**2), 2*c])
                    v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
                    a = l1*v1[0]**2 + l2*v2[0]**2
                    b = l1*v1[1]**2 + l2*v2[1]**2
                    c = l1*v1[0]*v1[1] + l2*v2[0]*v2[1]
                A.data[:, elem_ind] = np.array([a, b, c])

        if (count > 0):
            logging.info("\tProjection performed, " +
                         "{:.2%} elems have been shifted.".
                         format(count / self.coarse_mesh.elem_count))

    def obj_func_val(self, A: Variable):
        val = 0.0
        stiff_A = get_stiffness_matrix_opt(A)
        self.__GA_data = np.zeros(
            (self.homo_solution_ref_count, self.coarse_mesh.inner_node_count))
        for homo_ind in range(self.homo_solution_ref_count):
            u_coarse, info = SPLSolver(stiff_A, self.rhs_data[homo_ind, :])
            assert (info == 0), "The solver fails!"
            self.__GA_data[homo_ind, :] = u_coarse
            diff = self.homo_solution_ref_data[homo_ind, :]-u_coarse
            val += (stiff_A @ diff) @ diff
        val = val + self.rho * self.get_norm2_square_coeff(A)
        return val

    def grad_obj_func_val(self, A: Variable):
        grad = np.zeros(A.data.shape)
        for homo_ind in range(self.homo_solution_ref_count):
            grad += (self.get_u2coeff(self.homo_solution_ref_data[homo_ind, :]) -
                     self.get_u2coeff(self.__GA_data[homo_ind, :]))
        grad = grad + 2.0 * self.rho * A.data
        return grad

    def opt_routine(self, A0: Variable, gamma: float = 0.5):
        A0_norm2 = np.sqrt(self.get_norm2_square_coeff(A0))
        A_curr = A0
        obj_val_curr = self.obj_func_val(A_curr)
        for opt_step_ind in range(EllipticCoeffOpt.MAX_OPT_STEPS):
            start = time.time()
            # negative gradient direction
            B = -1.0 * self.grad_obj_func_val(A_curr)
            sigma = 1.0
            for line_search_ind in range(EllipticCoeffOpt.MAX_LINE_SEARCH):
                A_next = A_curr.init_by_copy()
                A_next.data = A_curr.data + B
                self.project_to_feasible_set(A_next)
                obj_val_next = self.obj_func_val(A_next)
                # Armijo rule
                temp = self.rho / sigma * self.get_norm2_square_coeff_data(B)
                if (obj_val_curr - obj_val_next > temp):
                    break
                else:
                    sigma = sigma / 2.0
                    B = B * sigma

            if (line_search_ind == EllipticCoeffOpt.MAX_LINE_SEARCH - 1):
                logging.critical("Line search reaches max iteration, careful!")
                A_next = A_curr
                obj_val_next = obj_val_curr

            descent = np.sqrt(self.get_norm2_square_coeff_data(
                A_next.data-A_curr.data))
            if (descent < EllipticCoeffOpt.EPSILON * A0_norm2):
                logging.info("Optimization process ends successfully.")
                # RETURN
                return A_next, obj_val_next
            elif (opt_step_ind == EllipticCoeffOpt.MAX_OPT_STEPS - 1):
                logging.critical("Optimization reaches max steps, careful!")
                # RETURN
                return A_next, obj_val_next
            else:
                A_curr = A_next
                obj_val_curr = obj_val_next
                end = time.time()
                logging.info("Optimization step index=%d, " +
                             "current object value=%.5f, " +
                             "opt cycle consuming time=%.3f.",
                             opt_step_ind, obj_val_next, end-start)
