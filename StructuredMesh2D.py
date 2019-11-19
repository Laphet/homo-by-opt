import numpy as np
import time
from scipy.sparse import csr_matrix, lil_matrix
import logging
from typing import Callable, Tuple
from numba import vectorize
from numba import int32, float64
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import cm
from matplotlib import colors
from matplotlib.ticker import MaxNLocator
import math

# rc('text', usetex=True)


class Mesh:
    NODE_COUNT_PER_ELEM = 4
    NODE_COUNT_PER_EDGE = 2
    ELEM_STIFFNESS_XX = np.array([[0.333333, 0.166667, -0.333333, -0.166667],
                                  [0.166667, 0.333333, -0.166667, -0.333333],
                                  [-0.333333, -0.166667, 0.333333, 0.166667],
                                  [-0.166667, -0.333333, 0.166667, 0.333333]])
    ELEM_STIFFNESS_YY = np.array([[0.333333, -0.333333, 0.166667, -0.166667],
                                  [-0.333333, 0.333333, -0.166667, 0.166667],
                                  [0.166667, -0.166667, 0.333333, -0.333333],
                                  [-0.166667, 0.166667, -0.333333, 0.333333]])
    ELEM_STIFFNESS_XY = np.array([[0.25, -0.25, 0.25, -0.25],
                                  [0.25, -0.25, 0.25, -0.25],
                                  [-0.25, 0.25, -0.25, 0.25],
                                  [-0.25, 0.25, -0.25, 0.25]])
    ELEM_STIFFNESS_YX = np.array([[0.25, 0.25, -0.25, -0.25],
                                  [-0.25, -0.25, 0.25, 0.25],
                                  [0.25, 0.25, -0.25, -0.25],
                                  [-0.25, -0.25, 0.25, 0.25]])
    ELEM_STIFFNESS_XYPYX = ELEM_STIFFNESS_XY + ELEM_STIFFNESS_YX
    ELEM_STIFFNESS_XXPYY = ELEM_STIFFNESS_XX + ELEM_STIFFNESS_YY
    ELEM_MASS_FF = np.array([[0.444444, 0.222222, 0.222222, 0.111111],
                             [0.222222, 0.444444, 0.111111, 0.222222],
                             [0.222222, 0.111111, 0.444444, 0.222222],
                             [0.111111, 0.222222, 0.222222, 0.444444]])
    ELEM_MASS_FZ = 1.0

    def __init__(self, _M: int):
        assert (_M > 1)
        self.M = _M
        self.elem_count = _M * _M
        self.node_count = (_M+1) * (_M+1)
        self.inner_node_count = (_M-1) * (_M-1)
        self.boundary_node_count = 4 * _M

    def _get_elem_ind2D_(self, elem_ind: int):
        """
            elem_ind2D = (m, n)
            elem_ind = i
            i = m*M + n
        """
        assert (0 <= elem_ind < self.elem_count)
        return divmod(elem_ind, self.M)

    def _get_elem_ind_from_ind2D_(self, m: int, n: int):
        assert(0 <= m < self.M and 0 <= n < self.M)
        return m*self.M + n

    def _get_node_ind_from_ind2D_(self, ind_x: int, ind_y: int):
        return ind_x*(self.M+1) + ind_y

    def _get_inner_node_ind_from_ind2D_(self, i: int, j: int):
        if (i == 0 or j == 0 or i == self.M or j == self.M):
            return -1
        else:
            return (i-1) * (self.M-1) + j - 1

    def get_refined_mesh(self, refine_num: int):
        assert (refine_num > 1)
        return Mesh(self.M * refine_num)

    def get_node_ind(self, elem_ind: int, local_node_ind: int) -> int:
        assert (0 <= elem_ind < self.elem_count and
                0 <= local_node_ind < self.NODE_COUNT_PER_ELEM)
        m, n = self._get_elem_ind2D_(elem_ind)
        ii, jj = divmod(local_node_ind, self.NODE_COUNT_PER_EDGE)
        return (m+ii) * (self.M+1) + n + jj

    def get_node_indR(self, local_node_ind: int, elem_ind: int) -> int:
        return self.get_node_ind(elem_ind, local_node_ind)

    def get_inner_node_ind(self, elem_ind: int, local_node_ind: int) -> int:
        """
            return: -1 if it is a boundary node
        """
        assert (0 <= elem_ind < self.elem_count and
                0 <= local_node_ind < self.NODE_COUNT_PER_ELEM)
        m, n = self._get_elem_ind2D_(elem_ind)
        ii, jj = divmod(local_node_ind, self.NODE_COUNT_PER_EDGE)
        i, j = m + ii, n + jj
        if (i == 0 or j == 0 or i == self.M or j == self.M):
            return -1
        else:
            return (i-1) * (self.M-1) + j - 1

    def get_inner_node_indR(self, local_node_ind: int, elem_ind: int) -> int:
        return self.get_inner_node_ind(elem_ind, local_node_ind)

    def get_node_coordinate(self, node_ind: int) -> Tuple[float, float]:
        m, n = divmod(node_ind, self.M+1)
        return float(m) / float(self.M), float(n) / float(self.M)

    def get_inner_node_coordinate(self, inner_node_ind: int) -> Tuple[float, float]:
        m, n = divmod(inner_node_ind, self.M-1)
        return float(m+1) / float(self.M), float(n+1) / float(self.M)

    def get_elem_barycenter_coordinate(self, elem_ind: int) -> Tuple[float, float]:
        m, n = self._get_elem_ind2D_(elem_ind)
        return (float(m)+0.5) / float(self.M), (float(n)+0.5) / float(self.M)

    def get_stiffness_trans_tensor(self):
        trans_tensor = np.zeros((self.inner_node_count, self.inner_node_count,
                                 Variable._MATRIX_ENTRY_PER_ELEM_ * self.elem_count))
        for T in range(self.elem_count):
            for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
                for beta in range(Mesh.NODE_COUNT_PER_ELEM):
                    i = self.get_inner_node_indR(alpha, T)
                    j = self.get_inner_node_indR(beta, T)
                    if (i >= 0 and j >= 0):
                        trans_tensor[i, j, Variable._MATRIX_ENTRY_PER_ELEM_ * T+0] = \
                            Mesh.ELEM_STIFFNESS_XX[alpha, beta]
                        trans_tensor[i, j, Variable._MATRIX_ENTRY_PER_ELEM_ * T+1] = \
                            Mesh.ELEM_STIFFNESS_YY[alpha, beta]
                        trans_tensor[i, j, Variable._MATRIX_ENTRY_PER_ELEM_ * T+2] = \
                            Mesh.ELEM_STIFFNESS_XYPYX[alpha, beta]

        return trans_tensor


class Variable:
    TYPE_DIC = {"zero-order": 0,
                "first-order": 1,
                "first-order-zeroBC": 2,
                "zero-order-matrix": 3}
    _MATRIX_ENTRY_PER_ELEM_ = 3

    def __init__(self, _mesh: Mesh, _type: int):
        assert (_type in self.TYPE_DIC.values())
        self.type = _type
        self.mesh = _mesh
        self.data = None
        if (_type == self.TYPE_DIC["zero-order"]):
            self.data = np.zeros((_mesh.elem_count, ))
        elif(_type == self.TYPE_DIC["first-order"]):
            self.data = np.zeros((_mesh.node_count, ))
        elif(_type == self.TYPE_DIC["first-order-zeroBC"]):
            self.data = np.zeros((_mesh.inner_node_count))
        elif(_type == self.TYPE_DIC["zero-order-matrix"]):
            self.data = np.zeros(
                (self._MATRIX_ENTRY_PER_ELEM_, _mesh.elem_count))
        else:
            logging.error("You should not arrive here.")

    @staticmethod
    def _get_eigens_(abc: np.array):
        assert(abc.shape == (Variable._MATRIX_ENTRY_PER_ELEM_, ))
        m = 0.5 * (abc[0]+abc[1])
        n = 0.5 * np.sqrt(np.abs((abc[0]-abc[1])**2+4.0*abc[2]**2))
        return np.array([m-n, m+n])

    def get_min_eigen(self):
        if (self.type == Variable.TYPE_DIC["zero-order-matrix"]):
            min_eigens = np.zeros((self.mesh.elem_count, ))
            for elem_ind in range(self.mesh.elem_count):
                min_eigens[elem_ind] = np.min(
                    Variable._get_eigens_(self.data[:, elem_ind]))
            return np.min(min_eigens)
        else:
            return -1.0

    def __add__(self, other: 'Variable') -> 'Variable':
        assert (self.type == other.type)
        M0, M1 = self.mesh.M, other.mesh.M
        g = math.gcd(M0, M1)
        r0, r1 = M1//g, M0//g

        var0 = self.project_to_refined_mesh(r0) if r0 > 1 else self
        var1 = other.project_to_refined_mesh(r1) if r1 > 1 else other
        assert (var0.mesh == var1.mesh)

        var0.data = var0.data + var1.data
        return var0

    def __sub__(self, other: 'Variable') -> 'Variable':
        other.data = -other.data
        return self.__add__(other)

    def direct_add(self, other):
        sum = self.init_by_copy()
        sum.data = self.data + other.data
        return sum

    def direct_div(self, scalar):
        self.data = self.data / scalar

    def diff_inf_norm(self, other):
        data = self.data - other.data
        return np.max(np.abs(data))

    def inf_norm(self):
        return np.max(np.abs(self.data))

    def init_by_copy(self):
        return Variable(self.mesh, self.type)

    def reset_data(self):
        self.data = np.zeros(self.data.shape)

    def check_positiveness(self, epsilon=1.0e-5):
        """
            If the coefficients (matrix) are positive (difined),
            return the minimal eigenvalue,
            Else return -1.0
        """
        if (self.type == self.TYPE_DIC["zero-order"]):
            _lambda = np.min(self.data)
        elif (self.type == self.TYPE_DIC["zero-order-matrix"]):
            eigens = np.array([Variable._get_eigens_(self.data[:, i])
                               for i in range(self.mesh.elem_count)])
            _lambda = np.min(eigens)
        else:
            _lambda = -1
        return _lambda if _lambda > epsilon else -1

    def evaluation_data_by_func(self, func,
                                func0=None,
                                func1=None,
                                func2=None,
                                args=None):
        if (self.type == Variable.TYPE_DIC["zero-order"]):
            for elem_ind in range(self.mesh.elem_count):
                self.data[elem_ind] = func(
                    *(self.mesh.get_elem_barycenter_coordinate(elem_ind)), args)
        elif (self.type == Variable.TYPE_DIC["first-order"]):
            for node_ind in range(self.mesh.node_count):
                self.data[node_ind] = func(
                    *(self.mesh.get_node_coordinate(node_ind)), args)
        elif (self.type == Variable.TYPE_DIC["first-order-zeroBC"]):
            for inner_node_ind in range(self.mesh.inner_node_count):
                self.data[inner_node_ind] = func(
                    *(self.mesh.get_inner_node_coordinate(inner_node_ind)), args)
        elif (self.type == Variable.TYPE_DIC["zero-order-matrix"]):
            assert (func0 != None and func1 != None and func2 != None)
            for elem_ind in range(self.mesh.elem_count):
                self.data[0, elem_ind] = func0(
                    *(self.mesh.get_elem_barycenter_coordinate(elem_ind)), args)
                self.data[1, elem_ind] = func1(
                    *(self.mesh.get_elem_barycenter_coordinate(elem_ind)), args)
                self.data[2, elem_ind] = func2(
                    *(self.mesh.get_elem_barycenter_coordinate(elem_ind)), args)
        else:
            logging.error("You should not arrive here.")
            assert False

    def project_to_refined_mesh(self, refine_num: int):
        r_mesh = self.mesh.get_refined_mesh(refine_num)
        r_var = Variable(r_mesh, self.type)
        if (self.type == Variable.TYPE_DIC["zero-order"]):
            for r_elem_ind in range(r_mesh.elem_count):
                r_m, r_n = r_mesh._get_elem_ind2D_(r_elem_ind)
                m, n = r_m // refine_num, r_n // refine_num
                r_var.data[r_elem_ind] = \
                    self.data[self.mesh._get_elem_ind_from_ind2D_(m, n)]

        elif (self.type == Variable.TYPE_DIC["zero-order-matrix"]):
            for r_elem_ind in range(r_mesh.elem_count):
                r_m, r_n = r_mesh._get_elem_ind2D_(r_elem_ind)
                m, n = r_m // refine_num, r_n // refine_num
                r_var.data[:, r_elem_ind] = \
                    self.data[:, self.mesh._get_elem_ind_from_ind2D_(m, n)]

        elif (self.type == Variable.TYPE_DIC["first-order"]):
            for r_node_ind in range(r_mesh.node_count):
                r_ind_x, r_ind_y = divmod(r_node_ind, r_mesh.M+1)
                # [0, r*self.M]
                ind_x, ind_y = r_ind_x // refine_num, r_ind_y // refine_num
                # [0, self.M)
                ind_x = ind_x if ind_x < self.mesh.M else ind_x - 1
                # [0, self.M)
                ind_y = ind_y if ind_y < self.mesh.M else ind_y - 1
                (u0, u1, u2, u3) = (
                    self.data[self.mesh._get_node_ind_from_ind2D_(
                        ind_x, ind_y)],
                    self.data[self.mesh._get_node_ind_from_ind2D_(
                        ind_x, ind_y+1)],
                    self.data[self.mesh._get_node_ind_from_ind2D_(
                        ind_x+1, ind_y)],
                    self.data[self.mesh._get_node_ind_from_ind2D_(ind_x+1, ind_y+1)])
                rho_x = float(r_ind_x-refine_num*ind_x) / float(refine_num)
                rho_y = float(r_ind_y-refine_num*ind_y) / float(refine_num)
                _temp_val = u0*(1.0-rho_x)*(1.0-rho_y) + u1*(1.0-rho_x)*rho_y + \
                    u2*rho_x*(1-rho_y) + u3*rho_x*rho_y
                r_var.data[r_node_ind] = _temp_val

        elif (self.type == Variable.TYPE_DIC["first-order-zeroBC"]):
            for r_inner_node_ind in range(r_mesh.inner_node_count):
                r_ind_x, r_ind_y = divmod(r_inner_node_ind, r_mesh.M-1)
                # [0, r*self.M-2]
                r_ind_x, r_ind_y = r_ind_x+1, r_ind_y+1
                # [0, r*self.M-1]
                ind_x, ind_y = r_ind_x // refine_num, r_ind_y // refine_num
                # [0, self.M-1]
                u0 = 0.0 if self.mesh._get_inner_node_ind_from_ind2D_(ind_x, ind_y) < 0 \
                    else self.data[self.mesh._get_inner_node_ind_from_ind2D_(ind_x, ind_y)]
                u1 = 0.0 if self.mesh._get_inner_node_ind_from_ind2D_(ind_x, ind_y+1) < 0 \
                    else self.data[self.mesh._get_inner_node_ind_from_ind2D_(ind_x, ind_y+1)]
                u2 = 0.0 if self.mesh._get_inner_node_ind_from_ind2D_(ind_x+1, ind_y) < 0 \
                    else self.data[self.mesh._get_inner_node_ind_from_ind2D_(ind_x+1, ind_y)]
                u3 = 0.0 if self.mesh._get_inner_node_ind_from_ind2D_(ind_x+1, ind_y+1) < 0 \
                    else self.data[self.mesh._get_inner_node_ind_from_ind2D_(ind_x+1, ind_y+1)]
                rho_x = float(r_ind_x-refine_num*ind_x) / float(refine_num)
                rho_y = float(r_ind_y-refine_num*ind_y) / float(refine_num)
                _temp_val = u0*(1.0-rho_x)*(1.0-rho_y) + u1*(1.0-rho_x)*rho_y + \
                    u2*rho_x*(1-rho_y) + u3*rho_x*rho_y
                r_var.data[r_inner_node_ind] = _temp_val
        else:
            logging.info("You should not arrive here.")
            assert False

        return r_var

    def average_to_coarsen_mesh(self, coarse_mesh: Mesh):
        avg_var = Variable(coarse_mesh, self.type)
        assert (self.mesh.M % coarse_mesh.M == 0)
        refine_num = self.mesh.M // coarse_mesh.M
        if (self.type == Variable.TYPE_DIC["zero-order-matrix"]):
            A0 = self.data[0, :].reshape((self.mesh.M, -1))
            A1 = self.data[1, :].reshape((self.mesh.M, -1))
            A2 = self.data[2, :].reshape((self.mesh.M, -1))
            for elem_ind in range(coarse_mesh.elem_count):
                m, n = divmod(elem_ind, coarse_mesh.M)
                avg_var.data[0, elem_ind] = np.average(
                    A0[m*refine_num:(m+1)*refine_num, n*refine_num:(n+1)*refine_num])
                avg_var.data[1, elem_ind] = np.average(
                    A1[m*refine_num:(m+1)*refine_num, n*refine_num:(n+1)*refine_num])
                avg_var.data[2, elem_ind] = np.average(
                    A2[m*refine_num:(m+1)*refine_num, n*refine_num:(n+1)*refine_num])
        return avg_var

    def get_plot(self, filename: str = None, component: int = 0):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axis([0.0, 1.0, 0.0, 1.0])
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_aspect('equal', 'box')

        if (self.type == Variable.TYPE_DIC["zero-order"]):
            X, Y = np.meshgrid(np.linspace(0.0, 1.0, self.mesh.M+1),
                               np.linspace(0.0, 1.0, self.mesh.M+1), indexing="ij")
            _data = self.data.reshape((self.mesh.M, -1))
            _max, _min = np.max(_data), np.min(_data)
            im = ax.pcolormesh(X, Y, _data, cmap="viridis", alpha=0.8,
                               edgecolors='none', norm=colors.Normalize(_min, _max))
            fig.colorbar(im, ax=ax)
        elif (self.type == Variable.TYPE_DIC["first-order"]):
            X, Y = np.meshgrid(np.linspace(0.0, 1.0, self.mesh.M+1),
                               np.linspace(0.0, 1.0, self.mesh.M+1), indexing="ij")
            _data = self.data.reshape((self.mesh.M+1, -1))
            _max, _min = np.max(_data), np.min(_data)
            levels = MaxNLocator(nbins=15).tick_values(_min, _max)
            cf = ax.contourf(X, Y, _data, levels=levels,
                             cmap="viridis", vmin=_min, vmax=_max)
            fig.colorbar(cf, ax=ax)
        elif (self.type == Variable.TYPE_DIC["first-order-zeroBC"]):
            X, Y = np.meshgrid(np.linspace(0.0, 1.0, self.mesh.M+1),
                               np.linspace(0.0, 1.0, self.mesh.M+1), indexing="ij")
            _data = np.zeros((self.mesh.M+1, self.mesh.M+1))
            _data[1:-1, 1:-1] = self.data.reshape((self.mesh.M-1, -1))
            _max, _min = np.max(_data), np.min(_data)
            levels = MaxNLocator(nbins=15).tick_values(_min, _max)
            cf = ax.contourf(X, Y, _data, levels=levels,
                             cmap="viridis", vmin=_min, vmax=_max)
            fig.colorbar(cf, ax=ax)
        elif (self.type == Variable.TYPE_DIC["zero-order-matrix"]):
            X, Y = np.meshgrid(np.linspace(0.0, 1.0, self.mesh.M+1),
                               np.linspace(0.0, 1.0, self.mesh.M+1), indexing="ij")
            assert (0 <= component < Variable._MATRIX_ENTRY_PER_ELEM_)
            _data = self.data[component, :].reshape((self.mesh.M, -1))
            _max, _min = np.max(_data), np.min(_data)
            im = ax.pcolormesh(X, Y, _data, cmap="viridis", alpha=0.8,
                               edgecolors='none', norm=colors.Normalize(_min, _max))
            fig.colorbar(im, ax=ax)

        if (filename != None):
            try:
                fig.savefig(filename)
            except:
                logging.error(
                    "Can not save the plot to file, check the filename.")
            else:
                logging.info("The plot has been saved.")
        else:
            plt.show()


def get_stiffness_matrix(co: Variable) -> csr_matrix:
    """
        return: csr_matrix with size freedom_num * freedom_num
    """
    logging.info("Begin to construct stiffness matrix.")
    start = time.time()
    _flag = 0
    if (co.type == Variable.TYPE_DIC["zero-order"]):
        _flag = 1
    elif (co.type == Variable.TYPE_DIC["zero-order-matrix"]):
        _flag = 2
    else:
        logging.error("You should not arrive here.")
        assert (_flag != 0)

    s = lil_matrix((co.mesh.inner_node_count, co.mesh.inner_node_count))

    for elem_ind in range(co.mesh.elem_count):
        for ii in range(co.mesh.NODE_COUNT_PER_ELEM):
            for jj in range(co.mesh.NODE_COUNT_PER_ELEM):
                m = co.mesh.get_inner_node_ind(elem_ind, ii)
                n = co.mesh.get_inner_node_ind(elem_ind, jj)
                if (m >= 0 and n >= 0):
                    if (_flag == 1):
                        s[m, n] += co.data[elem_ind] * \
                            Mesh.ELEM_STIFFNESS_XXPYY[ii, jj]
                    elif (_flag == 2):
                        s[m, n] += \
                            co.data[0, elem_ind] * Mesh.ELEM_STIFFNESS_XX[ii, jj] + \
                            co.data[1, elem_ind] * Mesh.ELEM_STIFFNESS_YY[ii, jj] + \
                            co.data[2, elem_ind] * \
                            Mesh.ELEM_STIFFNESS_XYPYX[ii, jj]

    s = s.tocsr()
    end = time.time()
    logging.info(
        "Finish constructing stiffness matrix, consuming time=%.3fs, " +
        "matrix size=(%d, %d).", end-start, *(s.shape()))
    return s


def get_mass_matrix(mesh: Mesh, type0: int, type1: int) -> csr_matrix:
    """
        returned matrix size depends on variable types:
        - (first-order, first-order) size (node_count, node_count)
        - (first-order-zeroBC, first-order-zeroBC) size (inner_node_count, inner_node_count)
        - (first-order-zeroBC, first-order) size (inner_node_count, node_count)
        - (zero-order, first-order) size (elem_count, node_count)
        - (first-order-zeroBC, zero-order) size (inner_node_count, elem_count)
    """
    logging.info("Begin to construct mass matrix.")
    start = time.time()
    _flag = 0
    if (type0 == Variable.TYPE_DIC["first-order"] and
            type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 1
        s = lil_matrix((mesh.node_count, mesh.node_count))
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["first-order-zeroBC"]):
        _flag = 2
        s = lil_matrix((mesh.inner_node_count, mesh.inner_node_count))
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 3
        s = lil_matrix((mesh.inner_node_count, mesh.node_count))
    elif (type0 == Variable.TYPE_DIC["zero-order"] and
          type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 4
        s = lil_matrix((mesh.elem_count, mesh.node_count))
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["zero-order"]):
        _flag = 5
        s = lil_matrix((mesh.inner_node_count, mesh.elem_count))
    else:
        logging.error("You should not arrive here.")
        assert (s != 0)

    h = 1.0 / float(mesh.M)
    r = 0.25 * h * h
    for elem_ind in range(mesh.elem_count):
        for ii in range(mesh.NODE_COUNT_PER_ELEM):
            if (_flag == 4):
                m = elem_ind
                n = mesh.get_node_ind(elem_ind, ii)
                s[m, n] += r * Mesh.ELEM_MASS_FZ
                continue
            elif (_flag == 5):
                m = mesh.get_inner_node_ind(elem_ind, ii)
                n = elem_ind
                if (m >= 0):
                    s[m, n] += r * Mesh.ELEM_MASS_FZ
                continue

            for jj in range(mesh.NODE_COUNT_PER_ELEM):
                if (_flag == 1):
                    m = mesh.get_node_ind(elem_ind, ii)
                    n = mesh.get_node_ind(elem_ind, jj)
                    s[m, n] += r * Mesh.ELEM_MASS_FF[ii, jj]
                elif (_flag == 2):
                    m = mesh.get_inner_node_ind(elem_ind, ii)
                    n = mesh.get_inner_node_ind(elem_ind, jj)
                    if (m >= 0 and n >= 0):
                        s[m, n] += r * Mesh.ELEM_MASS_FF[ii, jj]
                elif (_flag == 3):
                    m = mesh.get_inner_node_ind(elem_ind, ii)
                    n = mesh.get_node_ind(elem_ind, jj)
                    if (m >= 0):
                        s[m, n] += r * Mesh.ELEM_MASS_FF[ii, jj]

    s = s.tocsr()
    end = time.time()
    logging.info(
        "Finish constructing mass matrix, consuming time=%.3fs, matrix size=(%d, %d)",
        end-start, *(s.shape))
    return s


@vectorize
# @np.vectorize
def _is_positive(a):
    return 1.0 if a >= 0 else 0.0


@vectorize
# @np.vectorize
def _replace_negative(a):
    return a if a >= 0 else 0


def get_stiffness_matrix_opt(co: Variable) -> csr_matrix:
    """
        See paper the doi:10.1007/s10543-015-0587-4
    """
    start = time.time()
    _flag = 0
    if (co.type == Variable.TYPE_DIC["zero-order"]):
        _flag = 1
    elif (co.type == Variable.TYPE_DIC["zero-order-matrix"]):
        _flag = 2
    else:
        logging.error("You should not arrive here.")
        assert (_flag != 0)

    s = csr_matrix((co.mesh.inner_node_count, co.mesh.inner_node_count))

    me = np.fromfunction(np.vectorize(co.mesh.get_inner_node_indR),
                         (Mesh.NODE_COUNT_PER_ELEM, co.mesh.elem_count), dtype=np.int32)
    for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
        for beta in range(alpha+1, Mesh.NODE_COUNT_PER_ELEM):
            if (_flag == 1):
                data = co.data * Mesh.ELEM_STIFFNESS_XXPYY[alpha, beta] * \
                    _is_positive(me[alpha, :]) * _is_positive(me[beta, :])

            elif (_flag == 2):
                data = (co.data[0, :]*Mesh.ELEM_STIFFNESS_XX[alpha, beta] +
                        co.data[1, :]*Mesh.ELEM_STIFFNESS_YY[alpha, beta] +
                        co.data[2, :]*Mesh.ELEM_STIFFNESS_XYPYX[alpha, beta]) * \
                    _is_positive(me[alpha, :]) * _is_positive(me[beta, :])

            row_ind = _replace_negative(me[alpha, :])
            col_ind = _replace_negative(me[beta, :])
            K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)

            s = s + K

    s = s + s.transpose()

    for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
        if (_flag == 1):
            data = co.data * \
                Mesh.ELEM_STIFFNESS_XXPYY[alpha, alpha] * \
                _is_positive(me[alpha, :])

        elif (_flag == 2):
            data = (co.data[0, :]*Mesh.ELEM_STIFFNESS_XX[alpha, alpha] +
                    co.data[1, :]*Mesh.ELEM_STIFFNESS_YY[alpha, alpha] +
                    co.data[2, :]*Mesh.ELEM_STIFFNESS_XYPYX[alpha, alpha]) * \
                _is_positive(me[alpha, :])

        row_ind = _replace_negative(me[alpha, :])
        K = csr_matrix((data, (row_ind, row_ind)), shape=s.shape)
        s = s + K

    end = time.time()
    logging.info(
        "Finish constructing stiffness matrix by OPTIMIZED method, " +
        "consuming time=%.3fs, matrix size=(%d, %d).", end-start, *(s.shape))
    return s


def get_mass_matrix_opt(mesh: Mesh, type0: int, type1: int) -> csr_matrix:
    """
        See the paper doi:10.1007/s10543-015-0587-4
    """
    start = time.time()
    _flag = 0
    if (type0 == Variable.TYPE_DIC["first-order"] and
            type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 1
        s = csr_matrix((mesh.node_count, mesh.node_count))
        me1 = np.fromfunction(np.vectorize(mesh.get_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["first-order-zeroBC"]):
        _flag = 2
        me2 = np.fromfunction(np.vectorize(mesh.get_inner_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
        s = csr_matrix((mesh.inner_node_count, mesh.inner_node_count))
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 3
        me1 = np.fromfunction(np.vectorize(mesh.get_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
        me2 = np.fromfunction(np.vectorize(mesh.get_inner_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
        s = csr_matrix((mesh.inner_node_count, mesh.node_count))
    elif (type0 == Variable.TYPE_DIC["zero-order"] and
          type1 == Variable.TYPE_DIC["first-order"]):
        _flag = 4
        me1 = np.fromfunction(np.vectorize(mesh.get_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
        s = csr_matrix((mesh.elem_count, mesh.node_count))
    elif (type0 == Variable.TYPE_DIC["first-order-zeroBC"] and
          type1 == Variable.TYPE_DIC["zero-order"]):
        _flag = 5
        me2 = np.fromfunction(np.vectorize(mesh.get_inner_node_indR),
                              (Mesh.NODE_COUNT_PER_ELEM, mesh.elem_count), dtype=np.int32)
        s = csr_matrix((mesh.inner_node_count, mesh.elem_count))
    else:
        logging.error("You should not arrive here.")
        assert (s != 0)

    h = 1.0 / float(mesh.M)
    r = 0.25 * h * h
    if (_flag == 1):
        for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
            for beta in range(Mesh.NODE_COUNT_PER_ELEM):
                data = r * Mesh.ELEM_MASS_FF[alpha, beta] * \
                    np.ones(mesh.elem_count)
                row_ind = me1[alpha, :]
                col_ind = me1[beta, :]
                K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)
                s = s + K
    elif (_flag == 2):
        for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
            for beta in range(Mesh.NODE_COUNT_PER_ELEM):
                data = r * Mesh.ELEM_MASS_FF[alpha, beta] * \
                    np.ones(mesh.elem_count) * \
                    _is_positive(me2[alpha, :]) * _is_positive(me2[beta, :])
                row_ind = _replace_negative(me2[alpha, :])
                col_ind = _replace_negative(me2[beta, :])
                K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)
                s = s + K
    elif (_flag == 3):
        for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
            for beta in range(Mesh.NODE_COUNT_PER_ELEM):
                data = r * Mesh.ELEM_MASS_FF[alpha, beta] * \
                    np.ones(mesh.elem_count) * _is_positive(me2[alpha, :])
                row_ind = _replace_negative(me2[alpha, :])
                col_ind = me1[beta, :]
                K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)
                s = s + K
    elif (_flag == 4):
        for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
            data = r * Mesh.ELEM_MASS_FZ * np.ones(mesh.elem_count)
            row_ind = np.arange(mesh.elem_count)
            col_ind = me1[alpha, :]
            K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)
            s = s + K
    elif (_flag == 5):
        for alpha in range(Mesh.NODE_COUNT_PER_ELEM):
            data = r * Mesh.ELEM_MASS_FZ * \
                np.ones(mesh.elem_count) * _is_positive(me2[alpha])
            row_ind = _replace_negative(me2[alpha, :])
            col_ind = np.arange(mesh.elem_count)
            K = csr_matrix((data, (row_ind, col_ind)), shape=s.shape)
            s = s + K

    end = time.time()
    logging.info(
        "Finish constructing mass matrix by OPTIMIZED method, " +
        "consuming time=%.3fs, matrix size=(%d, %d).", end-start, *(s.shape))
    return s


if (__name__ == "__main__"):
    coarse_mesh = Mesh(32)
    refine_mesh = coarse_mesh.get_refined_mesh(16)
    refine_co = Variable(refine_mesh, Variable.TYPE_DIC["zero-order-matrix"])
    def func0(x, y, args=None): return 3.0*x + y + 1.0
    def func1(x, y, args=None): return x*(1.0-y)
    def func2(x, y, args=None): return x*(1.0-y)
    refine_co.evaluation_data_by_func(
        None, func0=func0, func1=func1, func2=func2)
    refine_co.get_plot("fig/refine_co_comp1.png", component=1)
    coarse_co = refine_co.average_to_coarsen_mesh(coarse_mesh)
    coarse_co.get_plot("fig/coarse_co_comp1.png", component=1)
