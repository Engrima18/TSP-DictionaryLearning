import numpy as np
import networkx as nx
from numpy.linalg import matrix_rank
from scipy.linalg import qr


def get_adj_en(g):
    return nx.adjacency_matrix(g).todense()


def get_b1(a):
    g = nx.Graph(a)
    return (-1)*nx.incidence_matrix(g, oriented=True).todense()


def getCycles(A, max_len=np.inf):
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


def get_b2(a, p_max_len=np.inf):
    G = nx.Graph(a)
    E_list = list(G.edges)
    All_P = getCycles(a, p_max_len)
    cycles = [cycle + [cycle[0]] for cycle in All_P] 
    edge_index_map = {edge: i for i, edge in enumerate(E_list)}
    B2 = np.zeros((len(E_list), len(cycles)))

    for cycle_index, cycle in enumerate(cycles):
        for i in range(len(cycle) - 1):
            edge = (cycle[i], cycle[i + 1])
            edge_reversed = (cycle[i + 1], cycle[i])

            # Use edge indices from the map to avoid repeated searches
            if edge in edge_index_map:
                B2[edge_index_map[edge], cycle_index] = 1
            elif edge_reversed in edge_index_map:
                B2[edge_index_map[edge_reversed], cycle_index] = -1
                
    QR = qr(B2, pivoting=True)
    rank = matrix_rank(B2)
    B2 = B2[:, sorted(QR[2]):rank]
    
    return B2