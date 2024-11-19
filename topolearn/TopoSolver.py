import scipy.linalg as sla
import scipy.sparse.linalg as ssla
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from cvxpy.error import SolverError
from .data_generation import *
from .utilsTopoSolver import *
from .Hodgelets import *  # SeparateHodgelet, SimplicianSlepians
from .EnhancedGraph import EnhancedGraph
from typing import Tuple, List, Union, Dict
import pickle
from functools import wraps
from einops import rearrange, reduce


class TopoSolver:

    def __init__(self, X_train, X_test, Y_train, Y_test, *args, **kwargs):

        params = {
            "P": None,  # Number of Kernels (Sub-dictionaries)
            "J": None,  # Polynomial order
            "K0": None,  # Sparsity level
            "dictionary_type": None,
            "c": None,  # spectral control parameter
            "epsilon": None,  # spectral control parameter
            "n": 10,  # number of nodes
            "sub_size": None,  # Number of sub-sampled nodes
            "true_prob_T": 1.0,  # True ratio of colored triangles
            "prob_T": 1.0,  # The triangle probability with which we want to bias our topology
            "p_edges": 1.0,  # Probability of edge existence
            "G_true": None,
            "seed": None,  ####
            "option": "One-shot-diffusion",
            "diff_order_sol": 1,
            "diff_order_irr": 1,
            "step_prog": 1,
            "top_k_slepians": 10,
            "B1_true": None,
            "B2_true": None,
        }

        self.testing_trace = (
            {}
        )  ##################################################################

        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ValueError(
                    "When using positional arguments, must provide a single dictionary"
                )
            params.update(args[0])

        params.update(kwargs)

        # Data
        self.X_train: np.ndarray = X_train
        self.X_test: np.ndarray = X_test
        self.Y_train: np.ndarray = Y_train
        self.Y_test: np.ndarray = Y_test
        self.m_train: int = Y_train.shape[1]
        self.m_test: int = Y_test.shape[1]

        # Topology and geometry behind data (by default we consider a topology with full upper laplacian)
        if params["G_true"] == None:
            self.G = EnhancedGraph(
                n=params["n"],
                p_edges=params["p_edges"],
                p_triangles=params["prob_T"],
                seed=params["seed"],
            )
        # If we know the true topology we completely use it
        else:
            self.G = params["G_true"]

        if np.all(params["B1_true"] == None):

            # Incidence matrices
            self.B1: np.ndarray = self.G.get_b1()
            self.B2: np.ndarray = self.G.get_b2()

            # Sub-sampling if needed to decrease complexity
            if params["sub_size"] != None:
                self.B1 = self.B1[:, : params["sub_size"]]
                self.B2 = self.B2[: params["sub_size"], :]
                self.B2 = self.B2[:, np.sum(np.abs(self.B2), 0) == 3]
            # Laplacians according to the Hodge Theory for cell complexes
            Lu, Ld, L = self.G.get_laplacians(sub_size=params["sub_size"])
            self.Lu: np.ndarray = Lu  # Upper Laplacian
            self.Ld: np.ndarray = Ld  # Lower Laplacian
            self.L: np.ndarray = L  # Sum Laplacian
        else:
            self.B1 = params["B1_true"]
            self.B2 = params["B2_true"]
            self.Ld = (self.B1.T) @ self.B1
            self.Lu = self.B2 @ self.B2.T
            self.L = self.Lu + self.Ld

        # Topology dimensions and hyperparameters
        self.nu: int = self.B2.shape[1]
        self.nd: int = self.B1.shape[1]
        self.true_prob_T = params["true_prob_T"]
        self.T: int = int(np.ceil(self.nu * (1 - self.true_prob_T)))

        self.M = self.L.shape[0]
        self.dictionary_type = params["dictionary_type"]

        # Init the learning errors and error curve (history)
        self.min_error_train = 1e20
        self.min_error_test = 1e20
        self.train_history: List[np.ndarray] = []
        self.test_history: List[np.ndarray] = []
        self.opt_upper = 0

        # Dictionary hyperparameters
        self.P = params["P"]  # Number of sub-dicts
        self.J = params["J"]  # Polynomial order for the Hodge Laplacian

        # Assumed sparsity level
        self.K0 = params["K0"]
        self.q_star = int(np.ceil(np.ceil(0.05 * self.nu) + (self.K0 / 5 - 1)))
        # Init optimal values for sparse representations and overcomplete dictionary
        self.D_opt: np.ndarray = np.zeros((self.M, self.M * self.P))
        self.X_opt_train: np.ndarray = np.zeros(
            (self.M * self.P, self.Y_train.shape[1])
        )
        self.X_opt_test: np.ndarray = np.zeros((self.M * self.P, self.Y_test.shape[1]))

        ############################################################################################################
        ##                                                                                                        ##
        ##               This section is only for learnable (data-driven) dictionaries                            ##
        ##                                                                                                        ##
        ############################################################################################################

        # Initialize the optimal values of the dictionary coefficients
        self.zero_out_h()

        # Compute the polynomial extension for the Laplacians and the auxiliary
        # "pseudo-vandermonde" matrix for the constraints in the quadratic form
        if self.dictionary_type == "joint":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(
                self.L, self.J
            )
            self.B = compute_vandermonde(self.L, self.J).real
        elif self.dictionary_type == "edge":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(
                self.Ld, self.J
            )
            self.B = compute_vandermonde(self.Ld, self.J).real
        elif self.dictionary_type == "separated":
            self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(
                self.Lu, self.J, separated=True
            )
            self.Ldj, self.lambda_max_d_j, self.lambda_min_d_j = compute_Lj_and_lambdaj(
                self.Ld, self.J, separated=True
            )
            self.Bu = compute_vandermonde(self.Lu, self.J).real
            self.Bd = compute_vandermonde(self.Ld, self.J)[:, 1:].real
            self.B = np.hstack([self.Bu, self.Bd])

        # Auxiliary matrix to define quadratic form dor the dictionary learning step
        self.P_aux: np.ndarray = None
        # Flag variable: the dictionary is learnable or analytic
        self.dict_is_learnable = self.dictionary_type in [
            "separated",
            "joint",
            "edge",
        ]

        # Auxiliary tools for the Slepians-based dictionary setup
        if self.dictionary_type == "slepians":
            self.option = params["option"]
            self.diff_order_sol = params["diff_order_sol"]
            self.step_prog = params["step_prog"]
            self.diff_order_irr = params["diff_order_irr"]
            self.source_sol = np.ones((self.nd,))
            self.source_irr = np.ones((self.nd,))
            self.top_K_slepians = params["top_k_slepians"]
            self.spars_level = list(range(10, 80, 10))
            self.F_sol, self.F_irr = get_frequency_mask(
                self.B1, self.B2
            )  # Get frequency bands
            self.S_neigh, self.complete_coverage = cluster_on_neigh(
                self.B1,
                self.B2,
                self.diff_order_sol,
                self.diff_order_irr,
                self.source_sol,
                self.source_irr,
                self.option,
                self.step_prog,
            )
            self.R = [self.F_sol, self.F_irr]
            self.S = self.S_neigh

        # Auxiliary tools for the Wavelet-based dictionary setup
        elif self.dictionary_type == "wavelet":
            # Remember that this part should be updated if B2 or Lu are updated!
            self.w1 = np.linalg.eigvalsh(self.Lu)
            self.w2 = np.linalg.eigvalsh(self.Ld)

        if self.dict_is_learnable:
            # Hyperparameters for dictionary stability in frequency domain
            if params["c"] != None:
                self.c = params["c"]
                self.epsilon = params["epsilon"]
            else:
                self.spectral_control_params()

    def update_Lu(self, Lu_new):
        self.Lu = Lu_new
        self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(
            self.Lu, self.J, separated=True
        )
        self.Bu = compute_vandermonde(self.Lu, self.J).real
        self.B = np.hstack([self.Bu, self.Bd])

    # def init_hu(self):
    #     """
    #     Initialize the dictionary coefficients corresponding to the Upper Laplacian
    #     to avoid ill-initialization during the learning step of the Upper Laplacian
    #     in 'pessimistic' mode
    #     """
    #     # If find bad intialization, i.e. all zeros we need some action
    #     if np.sum(np.abs(self.h_opt[1])) == 0:
    #         # Check if the mean of the coefficients for the Lower Laplacian is a good candidate
    #         init_val = np.mean(self.h_opt[2])
    #         if init_val == 0:
    #             self.h_opt[1] = np.full((self.P, self.J), np.max(self.h_opt[2]))
    #         else:
    #             self.h_opt[1] = np.full((self.P, self.J), init_val)

    def zero_out_h(self):
        # Init the dictionary parameters according to the specific parameterization setup
        if self.dictionary_type == "separated":
            hs = np.zeros(
                (self.P, self.J)
            )  # multiplicative coefficients for Upper Laplacian
            hi = np.zeros(
                (self.P, self.J)
            )  # multiplicative coefficients for Lower Laplacian
            hh = np.zeros(
                (self.P, 1)
            )  # multiplicative coefficients for identity matrix
            self.h_opt: List[np.ndarray] = [hh, hs, hi]
        else:
            h = np.zeros((self.P, self.J))
            hi = np.zeros((self.P, 1))
            self.h_opt: List[np.ndarray] = [h, hi]

    def default_solver(self, solver, prob, solver_params={}):
        self.init_dict(mode="only_X")
        prob.solve(solver=solver, **solver_params)

    @staticmethod
    def _multiplier_search(*arrays, P, c, epsilon):
        is_okay = 0
        mult = 100
        tries = 0
        while is_okay == 0:
            is_okay = 1
            h, c_try, _, tmp_sum_min, tmp_sum_max = generate_coeffs(
                arrays, P=P, mult=mult
            )
            if c_try <= c:
                is_okay *= 1
            if tmp_sum_min > c - epsilon:
                is_okay *= 1
                incr_mult = 0
            else:
                is_okay = is_okay * 0
                incr_mult = 1
            if tmp_sum_max < c + epsilon:
                is_okay *= 1
                decr_mult = 0
            else:
                is_okay *= 0
                decr_mult = 1
            if is_okay == 0:
                tries += 1
            if tries > 3:
                discard = 1
                break
            if incr_mult == 1:
                mult *= 2
            if decr_mult == 1:
                mult /= 2
        return h, discard

    def init_dict(
        self, h_prior: np.ndarray = None, mode: str = "only_X"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize the dictionary and the signal sparse representation for the alternating
        optimization algorithm.

        Args:
            Lu (np.ndarray): Upper Laplacian matrix
            Ld (np.ndarray): Lower Laplacian matrix
            P (int): Number of kernels (sub-dictionaries).
            J (int): Max order of the polynomial for the single sub-dictionary.
            Y_train (np.ndarray): Training data.
            K0 (int): Sparsity of the signal representation.
            dictionary_type (str): Type of dictionary.
            c (float): Boundary constant from the synthetic data generation process.
            epsilon (float): Boundary constant from the synthetic data generation process.
            only (str): Type of initialization. Can be one of: "only_X", "all", "only_D".

        Returns:
            Tuple[np.ndarray, np.ndarray, bool]: Initialized dictionary, initialized sparse representation, and discard flag value.
        """
        self.min_error_train, self.min_error_test = 1e20, 1e20
        self.zero_out_h()

        # If no prior info on the dictionary
        if np.all(h_prior == None):

            # Init Dictionary
            if mode in ["all", "only_D"]:

                discard = 1
                while discard == 1:

                    if self.dictionary_type != "separated":
                        h_prior, discard = self._multiplier_search(
                            self.lambda_max_j,
                            self.lambda_min_j,
                            P=self.P,
                            c=self.c,
                            epsilon=self.epsilon,
                        )
                        self.D_opt = generate_dictionary(h_prior, self.P, self.Lj)

                    else:
                        h_prior, discard = self._multiplier_search(
                            self.lambda_max_d_j,
                            self.lambda_min_d_j,
                            self.lambda_max_u_j,
                            self.lambda_min_u_j,
                            P=self.P,
                            c=self.c,
                            epsilon=self.epsilon,
                        )
                        self.D_opt = generate_dictionary(
                            h_prior, self.P, self.Luj, self.Ldj
                        )

            # Init Sparse Representations
            if mode in ["all", "only_X"]:

                L = self.Ld if self.dictionary_type == "edge" else self.L
                _, Dx = sla.eig(L)
                dd = la.norm(Dx, axis=0)
                W = np.diag(1.0 / dd)
                Dx = Dx / la.norm(Dx)
                Domp = Dx @ W
                X = np.apply_along_axis(
                    lambda x: get_omp_coeff(self.K0, Domp.real, x),
                    axis=0,
                    arr=self.Y_train,
                )
                X = np.tile(X, (self.P, 1))
                self.X_opt_train = X

        # Otherwise use prior info about the dictionary to initialize both the dictionary and the sparse representation
        else:

            self.h_opt = h_prior

            if self.dictionary_type == "separated":
                self.D_opt = generate_dictionary(h_prior, self.P, self.Luj, self.Ldj)
                self.X_opt_train = sparse_transform(self.D_opt, self.K0, self.Y_train)
            else:
                self.D_opt = generate_dictionary(h_prior, self.P, self.Lj)
                self.X_opt_train = sparse_transform(self.D_opt, self.K0, self.Y_train)

    def topological_dictionary_learn(
        self,
        lambda_: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        solver: str = "MOSEK",
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Dictionary learning algorithm implementation for sparse representations of a signal on complex regular cellular.
        The algorithm consists of an iterative alternating optimization procedure defined in two steps: the positive semi-definite programming step
        for obtaining the coefficients and dictionary based on Hodge theory, and the Orthogonal Matching Pursuit step for constructing
        the K0-sparse solution from the dictionary found in the previous step, which best approximates the original signal.
        Args:
            Y_train (np.ndarray): Training data.
            Y_test (np.ndarray): Testing data.
            J (int): Max order of the polynomial for the single sub-dictionary.
            M (int): Number of data points (number of nodes in the data graph).
            P (int): Number of kernels (sub-dictionaries).
            D0 (np.ndarray): Initial dictionary.
            X0 (np.ndarray): Initial sparse representation.
            Lu (np.ndarray): Upper Laplacian matrix
            Ld (np.ndarray): Lower Laplacian matrix
            dictionary_type (str): Type of dictionary.
            c (float): Boundary constant from the synthetic data generation process.
            epsilon (float): Boundary constant from the synthetic data generation process.
            K0 (int): Sparsity of the signal representation.
            lambda_ (float, optional): Regularization parameter. Defaults to 1e-3.
            max_iter (int, optional): Maximum number of iterations. Defaults to 10.
            patience (int, optional): Patience for early stopping. Defaults to 10.
            tol (float, optional): Tolerance value. Defaults to 1e-s.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            minimum training error, minimum testing error, optimal coefficients, optimal testing sparse representation, and optimal training sparse representation.
        """

        # Define hyperparameters
        iter_, pat_iter = 1, 0
        train_hist = []
        test_hist = []

        if self.dict_is_learnable:

            # Init the dictionary and the sparse representation
            D_coll = [
                cp.Constant(self.D_opt[:, (self.M * i) : (self.M * (i + 1))])
                for i in range(self.P)
            ]
            Dsum = cp.Constant(np.zeros((self.M, self.M)))
            h_opt = self.h_opt
            Y = cp.Constant(self.Y_train)
            X_tr = self.X_opt_train
            X_te = self.X_opt_test
            I = cp.Constant(np.eye(self.M))

            while pat_iter < patience and iter_ <= max_iter:

                # SDP Step
                X = cp.Constant(X_tr)
                if iter_ != 1:
                    D_coll = [
                        cp.Constant(D[:, (self.M * i) : (self.M * (i + 1))])
                        for i in range(self.P)
                    ]
                    Dsum = cp.Constant(np.zeros((self.M, self.M)))

                # Define the objective function
                if self.dictionary_type in ["joint", "edge"]:
                    # Init the variables
                    h = cp.Variable((self.P, self.J))
                    hI = cp.Variable((self.P, 1))
                    h.value, hI.value = h_opt
                    for i in range(0, self.P):
                        tmp = cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0, self.J):
                            tmp += cp.Constant(self.Lj[j, :, :]) * h[i, j]
                        tmp += I * hI[i]
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i] for i in range(self.P)])
                    term1 = cp.square(cp.norm((Y - D @ X), "fro"))
                    term2 = cp.square(cp.norm(h, "fro") * lambda_)
                    term3 = cp.square(cp.norm(hI, "fro") * lambda_)
                    obj = cp.Minimize(term1 + term2 + term3)

                else:
                    # Init the variables
                    hI = cp.Variable((self.P, self.J))
                    hS = cp.Variable((self.P, self.J))
                    hH = cp.Variable((self.P, 1))
                    hH.value, hS.value, hI.value = h_opt
                    for i in range(0, self.P):
                        tmp = cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0, self.J):
                            tmp += (cp.Constant(self.Luj[j, :, :]) * hS[i, j]) + (
                                cp.Constant(self.Ldj[j, :, :]) * hI[i, j]
                            )
                        tmp += I * hH[i]
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i] for i in range(self.P)])

                    term1 = cp.square(cp.norm((Y - D @ X), "fro"))
                    term2 = cp.square(cp.norm(hI, "fro") * lambda_)
                    term3 = cp.square(cp.norm(hS, "fro") * lambda_)
                    term4 = cp.square(cp.norm(hH, "fro") * lambda_)
                    obj = cp.Minimize(term1 + term2 + term3 + term4)

                # Define the constraints
                constraints = (
                    [D_coll[i] >> 0 for i in range(self.P)]
                    + [(cp.multiply(self.c, I) - D_coll[i]) >> 0 for i in range(self.P)]
                    + [
                        (Dsum - cp.multiply((self.c - self.epsilon), I)) >> 0,
                        (cp.multiply((self.c + self.epsilon), I) - Dsum) >> 0,
                    ]
                )

                prob = cp.Problem(obj, constraints)
                prob.solve(solver=eval(f"cp.{solver}"), verbose=False)

                # Dictionary Update
                D = D.value
                if self.dictionary_type in ["joint", "edge"]:
                    h_opt = [
                        h_opt[0] + step_h * (h.value - h_opt[0]),
                        h_opt[1] + step_h * (hI.value - h_opt[1]),
                    ]
                else:
                    h_opt = [
                        h_opt[0] + step_h * (hH.value - h_opt[0]),
                        h_opt[1] + step_h * (hS.value - h_opt[1]),
                        h_opt[2] + step_h * (hI.value - h_opt[2]),
                    ]

                # OMP Step
                X_te_tmp, X_tr_tmp = sparse_transform(
                    D, self.K0, self.Y_test, self.Y_train
                )
                # Sparse Representation Update
                X_tr = X_tr + step_x * (X_tr_tmp - X_tr)
                X_te = X_te + step_x * (X_te_tmp - X_te)

                # Error Update
                error_train = nmse(D, X_tr, self.Y_train, self.m_train)
                error_test = nmse(D, X_te, self.Y_test, self.m_test)
                train_hist.append(error_train)
                test_hist.append(error_test)

                # Error Storing
                if (
                    (error_train < self.min_error_train)
                    and (abs(error_train) > np.finfo(float).eps)
                    and (abs(error_train - self.min_error_train) > tol)
                ):
                    self.X_opt_train = X_tr
                    self.min_error_train = error_train

                if (
                    (error_test < self.min_error_test)
                    and (abs(error_test) > np.finfo(float).eps)
                    and (abs(error_test - self.min_error_test) > tol)
                ):
                    self.h_opt = h_opt
                    self.D_opt = D
                    self.X_opt_test = X_te
                    self.min_error_test = error_test
                    pat_iter = 0

                    if verbose == 1:
                        print("New Best Test Error:", self.min_error_test)
                else:
                    pat_iter += 1

                iter_ += 1

        else:

            # Fourier Dictionary Benchmark
            _, self.D_opt = sla.eigh(self.L)
            self.X_opt_test, self.X_opt_train = sparse_transform(
                self.D_opt, self.K0, self.Y_test, self.Y_train
            )

            # Error Updating
            self.min_error_train = nmse(
                self.D_opt, self.X_opt_train, self.Y_train, self.m_train
            )
            self.min_error_test = nmse(
                self.D_opt, self.X_opt_test, self.Y_test, self.m_test
            )

            train_hist.append(error_train)
            test_hist.append(error_test)

        return self.min_error_test, self.min_error_train, train_hist, test_hist

    def _aux_matrix_update(self, X):

        I = [np.eye(self.M)]
        if self.dictionary_type == "separated":
            # LLu = [lu for lu in self.Luj]
            # LLd = [ld for ld in self.Ldj]
            # LL = np.array(I + LLu + LLd)
            LL = np.concatenate((I, self.Luj, self.Ldj))
        else:
            # LL = [l for l in self.Lj]
            # LL = np.array(I + LL)
            LL = np.concatenate((I, self.Lj))
        self.P_aux = np.array(
            [LL @ X[(i * self.M) : ((i + 1) * self.M), :] for i in range(self.P)]
        )
        self.P_aux = rearrange(self.P_aux, "b h w c -> (b h) w c")

    def topological_dictionary_learn_qp(
        self,
        lambda_: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        solver: str = "GUROBI",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:

        # Define hyperparameters
        iter_, pat_iter = 1, 0
        train_hist = []
        test_hist = []
        solver = eval(f"cp.{solver}")

        # Learnable Dictionary -> alternating-direction optimization algorithm
        if self.dict_is_learnable:

            # Init the the sparse representation
            h_opt = np.hstack([h.flatten() for h in self.h_opt]).reshape(-1, 1)
            X_tr = self.X_opt_train
            X_te = self.X_opt_test
            f = 2 if self.dictionary_type == "separated" else 1
            reg = lambda_ * np.eye(self.P * (f * self.J + 1))
            I_s = cp.Constant(np.eye(self.P))
            i_s = cp.Constant(np.ones((self.P, 1)))
            B = cp.Constant(self.B.real)

            while pat_iter < patience and iter_ <= max_iter:
                # Init variables and parameters
                h = cp.Variable((self.P * (f * self.J + 1), 1))
                self._aux_matrix_update(X_tr)
                h.value = h_opt
                Q = cp.Constant(
                    np.einsum("imn, lmn -> il", self.P_aux, self.P_aux) + reg
                )
                l = cp.Constant(np.einsum("mn, imn -> i", self.Y_train, self.P_aux))

                # Quadratic term
                term2 = cp.quad_form(h, Q, assume_PSD=True)
                # Linear term
                term1 = l @ h
                term1 = cp.multiply(-2, term1)[0]

                obj = cp.Minimize(term2 + term1)

                # Define the constraints
                cons1 = cp.kron(I_s, B) @ h
                cons2 = cp.kron(i_s.T, B) @ h
                constraints = (
                    [cons1 >= 0]
                    + [cons1 <= self.c]
                    + [cons2 >= (self.c - self.epsilon)]
                    + [cons2 <= (self.c + self.epsilon)]
                )

                prob = cp.Problem(obj, constraints)

                solver_params = {
                    "NumericFocus": 1 | 2 | 3,
                    "Aggregate": 0,
                    "ScaleFlag": 2,
                    "ObjScale": -0.5,
                    "BarHomogeneous": 1,
                    "Method": 1,
                    "verbose": False,
                }
                try:
                    # If we are unable to move from starting conditions -> use default solver parameters
                    if pat_iter > 0 and np.all(h_opt == 0):
                        self.default_solver(solver, prob)
                    else:
                        prob.solve(solver=solver, **solver_params)
                        # If some solver parameters relax too much the problem -> use default solver parameters
                        if prob.status == "infeasible_or_unbounded":
                            self.default_solver(solver, prob)

                except SolverError:
                    solver_params = {"QCPDual": 0, "verbose": False}
                    # If in any case the solver with tuned parameters fails -> use the default solver parameters
                    self.default_solver(solver, prob, solver_params)

                # Update the dictionary
                if self.dictionary_type in ["joint", "edge"]:
                    h_list = rearrange_coeffs(h, self.J, self.P)
                    # h_tmp = h.value.reshape(self.P, self.J+1)
                    D = generate_dictionary(h_list, self.P, self.Lj)
                    h_opt = h_opt + step_h * (h.value - h_opt)
                else:
                    h_list = rearrange_coeffs(h, self.J, self.P, sep=True)
                    D = generate_dictionary(h_list, self.P, self.Luj, self.Ldj)
                    h_opt = h_opt + step_h * (h.value - h_opt)

                # OMP Step
                X_te_tmp, X_tr_tmp = sparse_transform(
                    D, self.K0, self.Y_test, self.Y_train
                )
                # Sparse Representation Update
                X_tr = X_tr + step_x * (X_tr_tmp - X_tr)
                X_te = X_te + step_x * (X_te_tmp - X_te)

                # Error Update
                error_train = nmse(D, X_tr, self.Y_train, self.m_train)
                error_test = nmse(D, X_te, self.Y_test, self.m_test)

                train_hist.append(error_train)
                test_hist.append(error_test)

                # Error Storing
                if (
                    (error_train < self.min_error_train)
                    and (abs(error_train) > np.finfo(float).eps)
                    and (abs(error_train - self.min_error_train) > tol)
                ):
                    self.X_opt_train = X_tr
                    self.min_error_train = error_train

                if (
                    (error_test < self.min_error_test)
                    and (abs(error_test) > np.finfo(float).eps)
                    and (abs(error_test - self.min_error_test) > tol)
                ):
                    self.h_opt = (
                        h_list if self.dictionary_type == "separated" else h_opt
                    )
                    self.D_opt = D
                    self.X_opt_test = X_te
                    self.min_error_test = error_test
                    pat_iter = 0

                    if verbose == 1:
                        print("New Best Test Error:", self.min_error_test)
                else:
                    pat_iter += 1

                iter_ += 1

        # Analytic dictionary directly go for the OMP step
        else:

            fit_intercept = True

            # Topological Fourier Dictionary
            if self.dictionary_type == "fourier":
                _, self.D_opt = sla.eig(self.L)

            # Classical Fourier Dictionary
            elif self.dictionary_type == "classic_fourier":
                self.D_opt = sla.dft(self.nd).real

            elif self.dictionary_type == "slepians":
                SS = SimplicianSlepians(
                    self.B1,
                    self.B2,
                    self.S,
                    self.R,
                    verbose=False,
                    top_K=self.top_K_slepians,
                )
                self.D_opt = SS.atoms_flat
                fit_intercept = False

            elif self.dictionary_type == "wavelet":
                SH = SeparateHodgelet(
                    self.B1,
                    self.B2,
                    *log_wavelet_kernels_gen(3, 4, np.log(np.max(self.w1))),
                    *log_wavelet_kernels_gen(3, 4, np.log(np.max(self.w2))),
                )
                self.D_opt = SH.atoms_flat
                fit_intercept = False
                # print(self.D_opt.shape)

            # OMP
            self.X_opt_test, self.X_opt_train = sparse_transform(
                self.D_opt, self.K0, self.Y_test, self.Y_train, fit_intercept
            )
            # Error Updating
            self.min_error_train = nmse(
                self.D_opt, self.X_opt_train, self.Y_train, self.m_train
            )
            self.min_error_test = nmse(
                self.D_opt, self.X_opt_test, self.Y_test, self.m_test
            )

            train_hist.append(self.min_error_train)
            test_hist.append(self.min_error_test)

        return self.min_error_test, self.min_error_train, train_hist, test_hist

    def mtv(self, gt_mask=None):
        """
        Min Total Variation algorithm, aimed to find the 'q' best candidate triangles
        inside our topology. The class uses this function to contrast the bad initialization problem
        of the coefficient related to the upper laplacian when applying the 'learn_upper_laplacian()'
        function in 'pessimistic' mode during joint dictionary and topology learning procedure.

        if 'gt_mask' is passed as an argument, the function checks that the selected candidate triangles
        are actually colored in the true topology.
        """
        assert (
            self.q_star < self.nu
        ), "The candidate number of triangles should be smaller than the max number of possible colored triangles in the topology"

        vals, Uirr = sla.eig(self.Ld)
        Uirr = Uirr[:, np.where(vals != 0)[0]]
        Ysh = np.eye(self.nd) - Uirr @ Uirr.T
        d = reduce((Ysh.T @ self.B2) ** 2, "m r -> r", "sum")
        q = np.argsort(d)[: self.q_star]

        if gt_mask != None:
            true_q = np.where(np.sum(gt_mask, axis=1) != 0)[0]
            checks = [q_i in true_q for q_i in q]
            checked = int(np.sum(checks))
            if checked < self.q_star:
                print(
                    f"Warning: {self.q_star-checked} of the {self.q_star} selected candidate triangles are not good!"
                )
            else:
                print("All the selected candidate triangles are good!")
        filter = np.zeros(self.nu)
        filter[q] = 1
        Lu_new = self.B2 @ np.diag(filter) @ self.B2.T
        self.update_Lu(Lu_new)
        return filter

    def learn_upper_laplacian(
        self,
        Lu_new: np.ndarray = None,
        filter: np.ndarray = 1,
        lambda_: float = 1e-3,
        max_iter: int = 10,
        patience: int = 10,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        mode: str = "optimistic",
        verbose: bool = False,
        warmup: int = 0,
        on_test: bool = False,
        QP=False,
    ):

        assert step_h < 1 or step_h > 0, "You must provide a step-size between 0 and 1."
        assert step_x < 1 or step_x > 0, "You must provide a step-size between 0 and 1."
        assert (mode == "optimistic") or (
            mode == "pessimistic"
        ), f'{mode} is not a legal mode: "optimistic" or "pessimistic" are the only ones allowed.'

        # Check if we are executing the first recursive iteration
        if np.all(Lu_new == None):
            T = self.B2.shape[1]
            self.warmup = warmup
            self.opt_upper = 0
            # start with a "full" upper Laplacian
            if mode == "optimistic":
                filter = np.ones(T)
            # start with an "empty" upper Laplacian
            elif mode == "pessimistic":
                filter = self.mtv()

        else:
            # if mode == "pessimistic":
            #     self.init_hu()
            self.update_Lu(Lu_new)

        if QP:

            try:
                _, _, train_hist, test_hist = self.topological_dictionary_learn_qp(
                    lambda_=lambda_,
                    max_iter=max_iter,
                    patience=patience,
                    tol=tol,
                    step_h=step_h,
                    step_x=step_x,
                    solver="GUROBI",
                )

            except SolverError:
                return (
                    self.min_error_train,
                    self.min_error_test,
                    self.train_history,
                    self.test_history,
                    self.Lu,
                    self.B2,
                )

        else:
            try:
                _, _, train_hist, test_hist = self.topological_dictionary_learn(
                    lambda_=lambda_,
                    max_iter=max_iter,
                    patience=patience,
                    tol=tol,
                    step_h=step_h,
                    step_x=step_x,
                )
            except SolverError:
                return (
                    self.min_error_train,
                    self.min_error_test,
                    self.train_history,
                    self.test_history,
                    self.Lu,
                    self.B2,
                )

        self.train_history.append(train_hist)
        self.test_history.append(test_hist)

        search_space = (
            np.where(filter == 1) if mode == "optimistic" else np.where(filter == 0)
        )
        sigmas = pd.DataFrame({"idx": search_space[0]})

        sigmas["sigma"] = sigmas.idx.apply(lambda _: filter)
        if mode == "optimistic":
            sigmas["sigma"] = sigmas.apply(lambda x: indicator_matrix(x), axis=1)
        else:
            sigmas["sigma"] = sigmas.apply(lambda x: indicator_matrix_rev(x), axis=1)
        sigmas["Luj"] = sigmas.apply(lambda x: compute_Luj(x, self.B2, self.J), axis=1)
        sigmas["D"] = sigmas.apply(
            lambda x: generate_dictionary(self.h_opt, self.P, x.Luj, self.Ldj), axis=1
        )
        if on_test:
            sigmas["X"] = sigmas.D.apply(
                lambda x: sparse_transform(x, self.K0, self.Y_test)
            )
            sigmas["NMSE"] = sigmas.apply(
                lambda x: nmse(x.D, x.X, self.Y_test, self.m_test), axis=1
            )
        else:
            sigmas["X"] = sigmas.D.apply(
                lambda x: sparse_transform(x, self.K0, self.Y_train)
            )
            sigmas["NMSE"] = sigmas.apply(
                lambda x: nmse(x.D, x.X, self.Y_train, self.m_train), axis=1
            )

        candidate_error = sigmas.NMSE.min()
        current_min = self.min_error_test if on_test else self.min_error_train
        idx_min = sigmas.NMSE.idxmin()
        self.testing_trace[f"{self.opt_upper}"] = (sigmas, current_min, candidate_error)
        # If in warmup look at the third decimal point
        if self.warmup > 0:
            candidate_error = int(candidate_error * 1000)
            current_min = int(current_min * 1000)
            self.warmup -= 1

        # self.testing_trace[f'{self.opt_upper}'] = (sigmas, current_min, candidate_error)
        if candidate_error <= current_min:
            S = sigmas.sigma[idx_min]
            Lu_new = self.B2 @ S @ self.B2.T
            filter = np.diagonal(S)
            self.opt_upper += 1

            if verbose:
                if mode == "optimistic":
                    print(
                        f"Removing {self.opt_upper} triangles from the topology... \n ... The min error: {candidate_error:.3f} !"
                    )
                else:
                    print(
                        f"Adding {self.opt_upper} triangles to the topology... \n ... The min error: {candidate_error:.3f} !"
                    )

            # self.testing_trace[f"{self.opt_upper}"] = (
            #     S,
            #     Lu_new,
            # )  # return the filter flattened matrix and the new best Lu_new

            return self.learn_upper_laplacian(
                Lu_new=Lu_new,
                filter=filter,
                lambda_=lambda_,
                max_iter=max_iter,
                patience=patience,
                tol=tol,
                step_h=step_h,
                step_x=step_x,
                mode=mode,
                verbose=verbose,
                on_test=on_test,
                QP=QP,
            )

        # if mode == "pessimistic":
        #     self.opt_upper = 0
        #     self.warmup = 1
        #     return self.learn_upper_laplacian(
        #         Lu_new=Lu_new,
        #         filter=filter,
        #         lambda_=lambda_,
        #         max_iter=max_iter,
        #         patience=patience,
        #         tol=tol,
        #         step_h=step_h,
        #         step_x=step_x,
        #         mode="optimistic",
        #         verbose=verbose,
        #         on_test=on_test,
        #         QP=QP,
        #     )

        self.B2 = self.B2 @ np.diag(filter)
        return (
            self.min_error_train,
            self.min_error_test,
            self.train_history,
            self.test_history,
            self.Lu,
            self.B2,
        )

    def save_results(func):
        """Decorator to save intermediate results when testing learning functions"""

        @wraps(func)
        def wrapper(self, *args, **kwargs):

            outputs = func(self, *args, **kwargs)
            func_name = func.__name__

            if func_name == "test_topological_dictionary_learn":

                path = os.getcwd()
                dir_path = os.path.join(
                    path, "results", "dictionary_learning", f"{self.dictionary_type}"
                )
                name = f"learn_D_{self.dictionary_type}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": outputs[2],
                    "test_history": outputs[3],
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            elif func_name == "test_topological_dictionary_learn_qp":

                path = os.getcwd()
                dir_path = os.path.join(path, "results", "no_topology_learning")
                name = f"learn_T{int(self.true_prob_T*100)}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": outputs[2],
                    "test_history": outputs[3],
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            elif func_name == "test_learn_upper_laplacian":

                path = os.getcwd()
                dir_path = os.path.join(path, "results", "topology_learning")
                name = f"learn_T{int(self.true_prob_T*100)}"
                filename = os.path.join(dir_path, f"{name}.pkl")
                save_var = {
                    "min_error_test": self.min_error_test,
                    "min_error_train": self.min_error_train,
                    "train_history": self.train_history,
                    "test_history": self.test_history,
                    "Lu_opt": self.Lu,
                    "B2_opt": self.B2,
                    "h_opt": self.h_opt,
                    "X_opt_test": self.X_opt_test,
                    "X_opt_train": self.X_opt_train,
                    "D_opt": self.D_opt,
                }

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            try:
                with open(filename, "wb") as file:
                    pickle.dump(save_var, file)
            except IOError as e:
                print(f"An error occurred while writing the file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            return outputs

        return wrapper

    @save_results
    def test_topological_dictionary_learn(
        self,
        mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        solver: str = "MOSEK",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
    ) -> None:

        try:
            self.init_dict(mode=mode)
        except:
            print("Initialization Failed!")

        self.topological_dictionary_learn(
            lambda_, max_iter, patience, tol, solver, step_h, step_x, verbose
        )

    @save_results
    def test_topological_dictionary_learn_qp(
        self,
        mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        solver: str = "GUROBI",
        step_h: float = 1.0,
        step_x: float = 1.0,
        verbose: bool = False,
    ) -> None:

        try:
            self.init_dict(mode=mode)
        except:
            print("Initialization Failed!")

        self.topological_dictionary_learn_qp(
            lambda_, max_iter, patience, tol, solver, step_h, step_x, verbose
        )

    @save_results
    def test_learn_upper_laplacian(
        self,
        init_mode: str = "only_X",
        lambda_: float = 1e-7,
        max_iter: int = 100,
        patience: int = 5,
        tol: float = 1e-7,
        step_h: float = 1.0,
        step_x: float = 1.0,
        mode: str = "optimistic",
        verbose: bool = True,
        warmup: int = 0,
        QP: bool = True,
    ) -> None:

        try:
            self.init_dict(mode=init_mode)
        except:
            print("Initialization Failed!")

        self.learn_upper_laplacian(
            lambda_=lambda_,
            max_iter=max_iter,
            patience=patience,
            tol=tol,
            step_h=step_h,
            step_x=step_x,
            mode=mode,
            verbose=verbose,
            warmup=warmup,
            QP=QP,
        )

    def get_topology_approx_error(self, Lu_true, round_res=None):
        res = np.linalg.norm(Lu_true - self.Lu, ord="fro")
        if round_res == None:
            return res
        return np.round(res, round_res)

    def get_test_error(self, round_res):
        if round_res == None:
            return self.min_error_test
        return np.round(self.min_error_test, round_res)

    def get_train_error(self, round_res=None):
        if round_res == None:
            return self.min_error_train
        return np.round(self.min_error_test, round_res)

    def get_numb_triangles(self, mode: str = None):
        if mode == "optimistic":
            return self.nu - self.opt_upper
        elif mode == "pessimistic":
            return self.nu + self.opt_upper
        return self.nu - self.T

    def fit(self, Lu_true, init_mode="only_X", learn_topology=True, **hyperparams):

        hp = {
            "lambda_": 1e-7,
            "tol": 1e-7,
            "patience": 5,
            "max_iter": 100,
            "step_x": 1.0,
            "step_h": 1.0,
            "QP": True,
            "mode": "optimistic",
            "verbose": False,
            "on_test": False,
        }

        hp.update(hyperparams)

        try:
            self.init_dict(mode=init_mode)
        except:
            print("Initialization Failed!")

        if learn_topology:
            self.learn_upper_laplacian(
                lambda_=hp["lambda_"],
                max_iter=hp["max_iter"],
                patience=hp["patience"],
                tol=hp["tol"],
                step_h=hp["step_h"],
                step_x=hp["step_x"],
                mode=hp["mode"],
                verbose=hp["verbose"],
                warmup=hp["warmup"],
                on_test=hp["on_test"],
                QP=hp["QP"],
            )

        else:
            self.topological_dictionary_learn_qp(
                lambda_=hp["lambda_"],
                max_iter=hp["max_iter"],
                patience=hp["patience"],
                tol=hp["tol"],
                step_h=hp["step_h"],
                step_x=hp["step_x"],
                solver="GUROBI",
            )

        Lu_approx_error = self.get_topology_approx_error(Lu_true=Lu_true)

        return self.min_error_train, self.min_error_test, Lu_approx_error

    def spectral_control_params(self, num=20, verbose=True):
        L = self.Ld if self.dictionary_type == "edge" else self.L
        vals, _ = sla.eig(L)
        c_in = np.sort(vals)[-1].real
        e_in = np.std(vals).real
        c_end = c_in**self.J
        e_end = e_in**self.J

        c_space = np.linspace(c_in, c_end, num=num)
        e_space = np.linspace(e_in, e_end, num=num)

        current_min = self.min_error_test
        # best_c = None
        # best_eps = None
        for e in e_space:
            for c in c_space:
                self.epsilon = e
                self.c = c
                self.init_dict(mode="only_X")
                self.topological_dictionary_learn_qp(lambda_=1e-7, max_iter=1)
                if self.min_error_test < current_min:
                    # best_c = self.c
                    # best_eps = self.e
                    current_min = self.min_error_test

        # self.c = best_c
        # self.epsilon = best_eps

        if verbose:
            print(f"Best c {self.c}")
            print(f"Best eps {self.epsilon}")
