import numpy as np
import networkx as nx
from numpy.linalg import matrix_rank
from scipy.linalg import qr
import pickle
import os
import hashlib


def memoize_to_pickle(file_name='cache\\DataTS.pkl'):
    '''
    Decorator for caching function results in a pickle file
    and speed up the calculation.
    '''
    def decorator(func):
        def wrapper(self, *args, **kwargs):

            graph_hash = hashlib.sha256(str((sorted(self.edges()), self.id)).encode()).hexdigest()
            key = f"{func.__name__}_{graph_hash}"
            
            if os.path.exists(file_name):
                with open(file_name, 'rb') as file:
                    try:
                        data = pickle.load(file)
                    except EOFError:
                        data = {}
                if key in data:
                    return data[key]

            result = func(self, *args, **kwargs)
            try:
                if os.path.exists(file_name):
                    with open(file_name, 'rb') as file:
                        data = pickle.load(file)
                else:
                    data = {}
            except EOFError:
                data = {}
            data[key] = result
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)

            return result
        return wrapper
    return decorator


class EnhancedGraph(nx.Graph):
    def __init__(self, n=10, p_edges=1., p_triangles=1., seed=None, *args, **kwargs):
        '''
        EnhancedGraph constructor that can generate an Erdős-Rényi graph.

        Parameters:
        - n (int): The number of nodes.
        - p (float): Probability for edge creation.
        - *args, **kwargs: Additional arguments passed to the nx.Graph constructor.
        '''

        super().__init__(*args, **kwargs)
        self.id = f'{seed}_{n}_{p_edges}'
        self.p_triangles = p_triangles
        er_graph = nx.erdos_renyi_graph(n, p_edges, seed)
        self.add_nodes_from(er_graph.nodes(data=True))
        self.add_edges_from(er_graph.edges(data=True))

        self.adjacency = nx.adjacency_matrix(self).todense()

    @memoize_to_pickle()
    def get_b1(self):
        '''
        Compute the oriented incidence matrix of the graph.
        '''
        return (-1) * nx.incidence_matrix(self, oriented=True).todense()

    def get_cycles(self, max_len=np.inf):
        '''
        Find all cycles in the graph within a specified maximum length.
        '''

        A = self.adjacency
        G = nx.DiGraph(A)
        cycles = nx.simple_cycles(G)

        seen = set()
        final = []

        for cycle in cycles:
            cycle_tuple = tuple(sorted(cycle))
            if cycle_tuple not in seen and 3 <= len(cycle) <= max_len:
                seen.add(cycle_tuple)
                final.append(cycle)

        final.sort(key=len)
        return final

    @memoize_to_pickle()
    def get_b2(self):
        '''
        Compute an edge-triangle incidence matrix with QR decomposition and rank considerations.
        '''
        E_list = list(self.edges)
        All_P = self.get_cycles()
        cycles = [cycle + [cycle[0]] for cycle in All_P]
        edge_index_map = {edge: i for i, edge in enumerate(E_list)}
        B2 = np.zeros((len(E_list), len(cycles)))

        for cycle_index, cycle in enumerate(cycles):
            for i in range(len(cycle) - 1):
                edge = (cycle[i], cycle[i + 1])
                edge_reversed = (cycle[i + 1], cycle[i])

                if edge in edge_index_map:
                    B2[edge_index_map[edge], cycle_index] = 1
                elif edge_reversed in edge_index_map:
                    B2[edge_index_map[edge_reversed], cycle_index] = -1

        QR = qr(B2, pivoting=True)
        rank = matrix_rank(B2)
        B2 = B2[:, QR[2][:rank]]

        return B2
    
    def get_laplacians(self, sub_size = None, full=False):
        
        B1 = self.get_b1()
        B2 = self.get_b2()

        # Sub-sampling if needed to decrease complexity
        if sub_size != None:
            B1 = B1[:, :sub_size]
            B2 = B2[:sub_size, :]
            
        B2 = B2[:,np.sum(np.abs(B2), 0) == 3]
        nu = B2.shape[1]

        if full:
            Lu = np.matmul(B2, np.transpose(B2), dtype=float)
            return Lu

        # Create a matrix to mask/color triangles
        prob_T = self.p_triangles # ratio of triangles that we want to retain from the original full topology
        T = int(np.ceil(nu*(1-prob_T)))
        mask = np.random.randint(0, nu, size=T)
        I_T = np.ones(nu)
        I_T[mask] = 0
        I_T = np.diag(I_T)

        # Laplacians
        Ld = np.matmul(np.transpose(B1), B1, dtype=float)
        Lu = B2@I_T@B2.T
        L = Lu+Ld

        return Lu, Ld, L