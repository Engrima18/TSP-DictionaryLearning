import scipy.linalg as sla
import numpy as np
import numpy.linalg as la
import cvxpy as cp
from .tsp_generation import *
from .EnhancedGraph import EnhancedGraph
from typing import Tuple, List, Union, Dict
import os
import pickle
from functools import wraps

def _indicator_matrix(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 0
    return np.diag(tmp)

def _indicator_matrix_rev(row):
    tmp = row.sigma.copy()
    tmp[row.idx] = 1
    return np.diag(tmp)

def _compute_Luj(row, b2, J):
    Lu = b2 @ row.sigma @ b2.T
    Luj = np.array([la.matrix_power(Lu, i) for i in range(1, J + 1)])
    return Luj

def sparse_transform(D, K0, Y_te, Y_tr=None):

    dd = la.norm(D, axis=0)
    W = np.diag(1. / dd)
    Domp = D @ W
    X_te = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_te)
    # Normalization
    X_te = W @ X_te

    if np.all(Y_tr == None):

        return X_te
    
    # Same for the training set
    X_tr = np.apply_along_axis(lambda x: get_omp_coeff(K0, Domp=Domp, col=x), axis=0, arr=Y_tr)
    X_tr = W @ X_tr
    
    return X_te, X_tr

def nmse(D, X, Y, m):
    return (1/m)* np.sum(la.norm(Y - (D @ X), axis=0)**2 /la.norm(Y, axis=0)**2)


class TspSolver:

    def __init__(self, X_train, X_test, Y_train, Y_test, *args, **kwargs):

        params = {
                'P': None,      # Number of Kernels (Sub-dictionaries)
                'J': None,      # Polynomial order
                'K0': None,     # Sparsity level
                'dictionary_type': None,
                'c': None,      # spectral control parameter 
                'epsilon': None,# spectral control parameter
                'n': 10,        # number of nodes
                'sub_size': None,   # Number of sub-sampled nodes
                'prob_T': 1.,   # Ratio of colored triangles
                'true_prob_T': 1.,   # True ratio of colored triangles
                'p_edges': 1.,  # Probability of edge existence
                'seed': None
                }
        
        if args:
            if len(args) != 1 or not isinstance(args[0], dict):
                raise ValueError("When using positional arguments, must provide a single dictionary")
            params.update(args[0])

        params.update(kwargs)

        # Data
        self.X_train: np.ndarray = X_train
        self.X_test: np.ndarray = X_test
        self.Y_train: np.ndarray = Y_train
        self.Y_test: np.ndarray = Y_test
        self.m_train: int = Y_train.shape[1]
        self.m_test: int = Y_test.shape[1]

        # Topology and geometry behind data
        self.G = EnhancedGraph(n=params['n'],
                               p_edges=params['p_edges'], 
                               p_triangles=params['prob_T'], 
                               seed=params['seed'])
        self.B1: np.ndarray = self.G.get_b1()
        self.B2: np.ndarray = self.G.get_b2()

        # Sub-sampling if needed to decrease complexity
        if params['sub_size'] != None:
            self.B1 = self.B1[:, :params['sub_size']]
            self.B2 = self.B2[:params['sub_size'], :]
            self.B2 = self.B2[:,np.sum(np.abs(self.B2), 0) == 3]
        self.nu: int = self.B2.shape[1]
        self.nd: int = self.B1.shape[1]
        self.true_prob_T = params['true_prob_T']
        self.T: int = int(np.ceil(self.nu*(1-params['prob_T'])))

        # Laplacians
        Lu, Ld, L = self.G.get_laplacians(sub_size=params['sub_size'])
        self.Lu: np.ndarray = Lu
        self.Ld: np.ndarray = Ld
        self.L: np.ndarray = L
        self.Lu_full: np.ndarray = G.get_laplacians(sub_size=params['sub_size'], 
                                                    full=True)
        self.M =  L.shape[0]
        self.history: List[np.ndarray] = []

        # Dictionary, parameters and hyperparameters for compression
        self.P = params['P']
        self.J = params['J']
        self.c = params['c']
        self.epsilon = params['epsilon']
        self.K0 = params['K0']
        self.dictionary_type = params['dictionary_type']
        self.D_opt: np.ndarray = np.zeros((self.M, self.M*self.P))

        if self.dictionary_type=="separated":
            hs = np.zeros((self.P,self.J))
            hi = np.zeros((self.P,self.J))
            hh = np.zeros((self.P,1))
            self.h_opt: List[np.ndarray] = [hs,hi,hh]
        else:
            self.h_opt: np.ndarray = np.zeros((self.P*(self.J+1),1))
            
        self.X_opt_train: np.ndarray = np.zeros(self.X_train.shape)
        self.X_opt_test: np.ndarray = np.zeros(self.X_test.shape)

        if self.dictionary_type == "joint":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(self.L, self.J)
        elif self.dictionary_type == "edge_laplacian":
            self.Lj, self.lambda_max_j, self.lambda_min_j = compute_Lj_and_lambdaj(self.Ld, self.J)
        elif  self.dictionary_type == 'separated':
            self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(self.Lu, self.J, separated=True)
            self.Ldj, self.lambda_max_d_j, self.lambda_min_d_j = compute_Lj_and_lambdaj(self.Ld, self.J, separated=True)

        # Init the learning errors
        self.min_error_train = 1e20
        self.min_error_test = 1e20

    # def fit(self) -> Tuple[float, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     min_error_test, _, _, h_opt, X_opt_test, D_opt = self.learn_upper_laplacian()
    #     return min_error_test, self.history, params['Lu'], h_opt, X_opt_test, D_opt

    def update_Lu(self, Lu_new):
        self.Lu = Lu_new
        self.Luj, self.lambda_max_u_j, self.lambda_min_u_j = compute_Lj_and_lambdaj(self.Lu, 
                                                                                    self.J, 
                                                                                    separated=True)

    @staticmethod
    def _multiplier_search(*arrays, P, c, epsilon):
        is_okay = 0
        mult = 100
        tries = 0
        while is_okay==0:
            is_okay = 1
            h, c_try, _, tmp_sum_min, tmp_sum_max = generate_coeffs(arrays, P=P, mult=mult)
            if c_try <= c:
                is_okay *= 1
            if tmp_sum_min > c-epsilon:
                is_okay *= 1
                incr_mult = 0
            else:
                is_okay = is_okay*0
                incr_mult = 1
            if tmp_sum_max < c+epsilon:
                is_okay *= 1
                decr_mult = 0
            else:
                is_okay *= 0
                decr_mult = 1
            if is_okay == 0:
                tries += 1
            if tries >3:
                discard = 1
                break
            if incr_mult == 1:
                mult *= 2
            if decr_mult == 1:
                mult /= 2
        return h, discard

    def init_dict(self,
                  h_prior: np.ndarray = None, 
                  mode: str = "only_X") -> Tuple[np.ndarray, np.ndarray]:
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
        
        # If no prior info on the dictionary
        if np.all(h_prior == None):

            if (mode in ["all","only_D"]):

                discard = 1
                while discard==1:

                    if dictionary_type != "separated":
                        h_prior, discard = _multiplier_search(self.lambda_max_j, 
                                                              self.lambda_min_j, 
                                                              P=self.P, 
                                                              c=self.c, 
                                                              epsilon=self.epsilon)
                        self.D_opt = generate_dictionary(h_prior, 
                                                         self.P, 
                                                         self.Lj)

                    else:
                        h_prior, discard = _multiplier_search(self.lambda_max_d_j, 
                                                              self.lambda_min_d_j, 
                                                              self.lambda_max_u_j, 
                                                              self.lambda_min_u_j,
                                                              P=self.P, 
                                                              c=self.c, 
                                                              epsilon=self.epsilon)
                        self.D_opt = generate_dictionary(h_prior, 
                                                         self.P, 
                                                         self.Luj, 
                                                         self.Ldj)
        
            if (mode in ["all","only_X"]):
                
                L = self.Ld if self.dictionary_type == "edge_laplacian" else self.L
                _, Dx = sla.eig(L)
                dd = la.norm(Dx, axis=0)
                W = np.diag(1./dd)
                Dx = Dx / la.norm(Dx)  
                Domp = Dx@W
                X = np.apply_along_axis(lambda x: get_omp_coeff(self.K0, Domp.real, x), axis=0, arr=self.Y_train)
                X = np.tile(X, (self.P,1))
                self.X_opt_train = X

        # Otherwise use prior info about the dictionary to initialize the sparse representation
        else:
            
            self.h_opt = h_prior

            if dictionary_type == "separated":
                self.D_opt = generate_dictionary(h_prior, 
                                                 self.P, 
                                                 self.Luj, 
                                                 self.Ldj)
                self.X_opt_train = sparse_transform(self.D_opt, 
                                                    self.K0, 
                                                    self.Y_train)
            else: 
                self.D_opt = generate_dictionary(h_prior, 
                                                 self.P, 
                                                 self.Lj)
                self.X_opt_train = sparse_transform(self.D_opt, 
                                                    self.K0, 
                                                    self.Y_train)             

   
    def save_results(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):

            outputs = func(self, *args, **kwargs)
            func_name = func.__name__

            if func_name == "topological_dictionary_learn":

                path = os.getcwd()
                dir_path = os.path.join(path, 
                                        'results', 
                                        'dictionary_learning',
                                        f'{self.dictionary_type}')
                name = f'learn_D_{self.dictionary_type}'
                filename = os.path.join(dir_path, f'{name}.pkl')
                save_var = {"min_error_test": self.min_error_test,
                            "min_error_train": self.min_error_train,
                            "history": outputs[2],
                            "h_opt": self.h_opt,
                            "X_opt_test": self.X_opt_test,
                            "X_opt_train": self.X_opt_train,
                            "D_opt": self.D_opt}
                
            elif func_name == "learn_upper_laplacian":

                path = os.getcwd()
                dir_path = os.path.join(path, 'results', 'topology_learning')
                name = f'learn_T{int(self.true_prob_T*100)}'
                filename = os.path.join(dir_path, f'{name}.pkl')
                save_var = {"min_error_test": self.min_error_test,
                            "min_error_train": self.min_error_train,
                            "history": self.history,
                            "Lu_opt": self.Lu,
                            "h_opt": self.h_opt,
                            "X_opt_test": self.X_opt_test,
                            "X_opt_train": self.X_opt_train,
                            "D_opt": self.D_opt}

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            try:
                with open(filename, 'wb') as file:
                    pickle.dump(save_var, file)
            except IOError as e:
                print(f"An error occurred while writing the file: {e}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            return outputs  
        return wrapper

    def topological_dictionary_learn(self,
                                     lambda_: float = 1e-3, 
                                     max_iter: int = 10, 
                                     patience: int = 10,
                                     tol: float = 1e-7,
                                     step_h: float = 1.,
                                     step_x: float = 1.,
                                     solver: str ="MOSEK", 
                                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray, List[float]]:
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
        hist = []

        if self.dictionary_type != "fourier":

            # Init the dictionary and the sparse representation 
            D_coll = [cp.Constant(self.D_opt[:,(self.M*i):(self.M*(i+1))]) for i in range(self.P)]
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
                    D_coll = [cp.Constant(D[:,(self.M*i):(self.M*(i+1))]) for i in range(self.P)]
                    Dsum = cp.Constant(np.zeros((self.M, self.M)))
                
                # Define the objective function
                if self.dictionary_type in ["joint", "edge_laplacian"]:
                    # Init the variables
                    h = cp.Variable((self.P, self.J))
                    h.value = h_opt
                    hI = cp.Variable((self.P, 1))
                    for i in range(0,self.P):
                        tmp =  cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0,self.J):
                            tmp += (cp.Constant(self.Lj[j, :, :]) * h[i,j])
                        tmp += (I*hI[i])
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i]for i in range(self.P)])
                    term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                    term2 = cp.square(cp.norm(h, 'fro')*lambda_)
                    term3 = cp.square(cp.norm(hI, 'fro')*lambda_)
                    obj = cp.Minimize(term1 + term2 + term3)

                else:
                    # Init the variables
                    hI = cp.Variable((self.P, self.J))
                    hS = cp.Variable((self.P, self.J))
                    hH = cp.Variable((self.P, 1))
                    hS.value, hI.value, hH.value = h_opt
                    for i in range(0,self.P):
                        tmp =  cp.Constant(np.zeros((self.M, self.M)))
                        for j in range(0,self.J):
                            tmp += ((cp.Constant(self.Luj[j, :, :])*hS[i,j]) + (cp.Constant(self.Ldj[j, :, :])*hI[i,j]))
                        tmp += (I*hH[i])
                        D_coll[i] = tmp
                        Dsum += tmp
                    D = cp.hstack([D_coll[i] for i in range(self.P)])
        
                    term1 = cp.square(cp.norm((Y - D @ X), 'fro'))
                    term2 = cp.square(cp.norm(hI, 'fro')*lambda_)
                    term3 = cp.square(cp.norm(hS, 'fro')*lambda_)
                    term4 = cp.square(cp.norm(hH, 'fro')*lambda_)
                    obj = cp.Minimize(term1 + term2 + term3 + term4)

                # Define the constraints
                constraints = [D_coll[i] >> 0 for i in range(self.P)] + \
                                [(cp.multiply(self.c, I) - D_coll[i]) >> 0 for i in range(self.P)] + \
                                [(Dsum - cp.multiply((self.c - self.epsilon), I)) >> 0, (cp.multiply((self.c + self.epsilon), I) - Dsum) >> 0]

                prob = cp.Problem(obj, constraints)
                prob.solve(solver=eval(f'cp.{solver}'), verbose=False)

                # Dictionary Update
                D = D.value
                if self.dictionary_type in ["joint", "edge_laplacian"]:
                    h_opt = h_opt + step_h*(h.value - h_opt)
                else:
                    h_opt = [h_opt[0] + step_h*(hS.value-h_opt[0]), 
                             h_opt[1] + step_h*(hI.value-h_opt[1]), 
                             h_opt[2] + step_h*(hH.value-h_opt[2])]

                # OMP Step
                X_te_tmp, X_tr_tmp = sparse_transform(D, self.K0, self.Y_test, self.Y_train)
                # Sparse Representation Update
                X_tr = X_tr + step_x*(X_tr_tmp - X_tr)
                X_te = X_te + step_x*(X_te_tmp - X_te)

                # Error Update
                error_train = nmse(D, X_tr, self.Y_train, self.m_train)
                error_test = nmse(D, X_te, self.Y_test, self.m_test)

                hist.append(error_test)
                
                # Error Storing
                if (error_train < self.min_error_train) and (abs(error_train) > np.finfo(float).eps) and (abs(error_train - self.min_error_train) > tol):
                    self.X_opt_train = X_tr
                    self.min_error_train = error_train

                if (error_test < self.min_error_test) and (abs(error_test) > np.finfo(float).eps) and (abs(error_test - self.min_error_test) > tol):
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
            self.X_opt_test, self.X_opt_train = sparse_transform(self.D_opt, self.K0, self.Y_test, self.Y_train)

            # Error Updating
            self.min_error_train = nmse(D, self.X_opt_train, self.Y_train, self.m_train)
            self.min_error_test= nmse(D, self.X_opt_test, self.Y_test, self.m_test)
            
        return self.min_error_test, self.min_error_train, hist
    
    @save_results
    def learn_upper_laplacian(self,
                              Lu_new: np.ndarray = None,
                              filter: np.ndarray = 1,
                              h_prior: np.ndarray = None,
                              lambda_: float = 1e-3, 
                              max_iter: int = 10, 
                              patience: int = 10,
                              tol: float = 1e-7,
                              step_h: float = 1.,
                              step_x: float = 1.,
                              mode: str = "optimistic",
                              verbose: bool = False):
    
        assert step_h<1 or step_h>0, "You must provide a step-size between 0 and 1."
        assert step_x<1 or step_x>0, "You must provide a step-size between 0 and 1."
        assert (mode=="optimistic") or (mode=="pessimistic"), f'{mode} is not a legal mode: \"optimistic\" or \"pessimistic\" are the only ones allowed.'
        
        # Check if we are executing the first recursive iteration
        if np.all(Lu_new == None):
            T = B2.shape[1]
            filter = np.ones(T) if mode=="optimistic" else np.zeros(T)
        else:
            self.update_Lu(Lu_new)

        self.init_dict(h_prior=h_prior,
                       mode="only_X")

        _, _, hist = self.topological_dictionary_learn(lambda_=lambda_,
                                                        max_iter=max_iter,
                                                        patience=patience,
                                                        tol=tol,
                                                        step_h=step_h,
                                                        step_x=step_x)
        self.history.append(hist)
        search_space = np.where(filter == 1) if mode=="optimistic" else np.where(filter == 0)   
        sigmas = pd.DataFrame({"idx": search_space[0]})

        sigmas["sigma"] = sigmas.idx.apply(lambda _: filter)
        if mode=="optimistic":
            sigmas["sigma"] = sigmas.apply(lambda x: _indicator_matrix(x), axis=1)
        else:
            sigmas["sigma"] = sigmas.apply(lambda x: _indicator_matrix_rev(x), axis=1)
        sigmas["Luj"] = sigmas.apply(lambda x: _compute_Luj(x, self.B2, self.J), axis=1)
        sigmas["D"] = sigmas.apply(lambda x: generate_dictionary(self.h_opt, self.P, x.Luj, self.Ldj), axis=1)
        sigmas["X"] = sigmas.D.apply(lambda x: sparse_transform(x, self.K0, self.Y_test))
        sigmas["NMSE"] = sigmas.apply(lambda x: nmse(x.D, x.X, self.Y_test, self.m_test), axis=1)
        
        candidate_error = sigmas.NMSE.min()
        idx_min = sigmas.NMSE.idxmin()

        if candidate_error < self.min_error_test:
            S = sigmas.sigma[idx_min]
            Lu_new = self.B2 @ S @ self.B2.T
            filter = np.diagonal(S)

            if verbose:
                print(f'Removing 1 triangle from topology... \n ... New min test error: {candidate_error} !')

            return self.learn_upper_laplacian(h_prior=self.h_opt,
                                              Lu_new=Lu_new,
                                              filter=filter,
                                              lambda_=lambda_,
                                              max_iter=max_iter,
                                              patience=patience,
                                              tol=tol,
                                              step_h=step_h,
                                              step_x=step_x,
                                              mode=mode,
                                              verbose=verbose)

        return self.min_error_test, self.history, self.Lu