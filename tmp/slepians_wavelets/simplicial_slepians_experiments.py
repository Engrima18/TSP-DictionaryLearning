# 2022/10/20~
# Claudio Battiloro, clabat@seas.upenn.edu/claudio.battiloro@uniroma1.it
# Paolo Di Lorenzo, paolo.dilorenzo@uniroma1.it

# Thanks to:
# Mitch Roddenberry (Rice ECE) for sharing the code from his paper:
# "Hodgelets: Localized Spectral Representations of Flows on Simplicial Complexes". 
# This code is built on top of it.


# This is the code used for implementing the numerical results in the paper:
# "Topological Slepians: Maximally localized representations of signals over simplicial complexes"
# C. Battiloro, P. Di Lorenzo, S. Barbarossa
# In particular, this code implements the vector-field representation task described in the paper.

from lib import *
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import sys
import time

res_dir = "/Users/Claudio/Desktop/Topological_Slepians/Results"

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# setup - data creation function #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

def hexgrid(n=15):

    xs = np.linspace(-2,2,n)
    ys = np.sqrt(3)*np.linspace(-2,2,n)
    
    xint = (xs[1]-xs[0])/2
    
    x1, y1 = np.meshgrid(xs,ys)
    x1[::2] += xint
    x1 = x1.flatten()
    y1 = y1.flatten()
    
    coords = np.vstack([x1,y1]).T

    # sort coords
    diagonal_coordinates = np.sum(coords, axis=1)  # y = -x + c, compute c
    diagonal_idxs = np.argsort(diagonal_coordinates)  # sort by c: origin comes first, upper-right comes last
    coords = coords[diagonal_idxs]  # apply sort to original coordinates
    
    tri = Delaunay(coords)
    simplices = [sorted(f) for f in tri.simplices]
    
    edges = []
    for f in simplices:
        [a,b,c] = sorted(f)
        edges.append((a,b))
        edges.append((b,c))
        edges.append((a,c))
    edges = sorted(set(edges))
    edges_tup = [tuple(e) for e in edges]

    edge_idx_dict = {edges_tup[i]: i for i in range(len(edges_tup))}

    nodes = list(range(len(coords)))

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    return graph, nodes, edges_tup, simplices, coords, edge_idx_dict

def incidence_matrices(G, V, E, faces, edge_to_idx):
    # redefined from nblib.py
    B1 = np.array(nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E),len(faces)])

    for f_idx, face in enumerate(faces): # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2

def discrete_flow(E, coords):

    flow_dict = {}

    A = np.array([[0.5, -1/(4*np.sqrt(3))],[1/(4*np.sqrt(3)), 0.5]])

    for e in E:
        j0,j1 = e
        p0 = coords[j0]
        p1 = coords[j1]

        clipradius = 0.7
        if (np.linalg.norm(0.5*(p0+p1)-np.array([np.pi/4,np.pi/4]))>clipradius) and \
           (np.linalg.norm(0.5*(p0+p1)-np.array([-np.pi/4,-np.pi/4]))>clipradius):
            flow_dict[e] = 0
            continue

        d = p1 - p0

        q0 = p0 + A @ d
        q1 = p0 + A.T @ d

        # black magic: F = [cos(x+y), sin(x-y)]
        # I integrate perpendicular to the edges between hexagons
        f = ((q0[1]-q1[1])/((q1[0]+q1[1])-(q0[0]+q0[1])) * (np.sin(q1[0]+q1[1])-np.sin(q0[0]+q0[1])) -
             (q0[0]-q1[0])/((q1[0]-q1[1])-(q0[0]-q0[1])) * (np.cos(q1[0]-q1[1])-np.cos(q0[0]-q0[1])))

        flow_dict[e] = f

    return flow_dict

#~~~~~~~~~~~~~~~#
# data building #
#~~~~~~~~~~~~~~~#
G, V, E, F, coords, edge_idx_dict = hexgrid()
N0 = len(V)
N1 = len(E)
N2 = len(F)
B1, B2 = incidence_matrices(G, V, E, F, edge_idx_dict)
B1, B2 = scale_incidence_matrices(B1, B2)
sgn_change = np.diag(np.sign(np.random.randn(N1)))
B1 = B1 @ sgn_change
B2 = sgn_change @ B2
L, L1, L2 = hodge_laplacians(B1, B2)

# Generate Signal
# compute a mapping from edges to the reals
flow_dict = discrete_flow(E, coords)
# then use the indexing to write as a vector in R^E
flow = np.zeros(N1)
for e in E:
    flow[edge_idx_dict[e]] = flow_dict[e]
flow /= np.linalg.norm(flow)

w = np.linalg.eigvalsh(L)
w1 = np.linalg.eigvalsh(L1)
w2 = np.linalg.eigvalsh(L2)

L_line = L.copy()
L_line = -(np.abs(L_line))
np.fill_diagonal(L_line, 0)
L_line -= np.diag(np.sum(L_line, axis=0))
w_line = np.linalg.eigvalsh(L_line)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Wavelet, Fourier and Slepians making #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

W_fourier = FourierBasis(B1, B2)
W_joint = JointHodgelet(B1, B2,
                        *log_wavelet_kernels_gen(3, 4, np.log(np.max(w))))
W_sep = SeparateHodgelet(B1, B2,
                         *log_wavelet_kernels_gen(3, 4, np.log(np.max(w1))),
                         *log_wavelet_kernels_gen(3, 4, np.log(np.max(w2))))
W_lift = LiftedHodgelet(B1, B2,
                        *log_wavelet_kernels_gen(3, 4, np.log(np.max(w1))),
                        *log_wavelet_kernels_gen(3, 4, np.log(np.max(w2))))
W_lift_mixed = MixedLiftedHodgelet(B1, B2,
                                   *log_wavelet_kernels_gen(3, 4, np.log(np.max(w1))),
                                   *log_wavelet_kernels_gen(3, 4, np.log(np.max(w2))))

W_fourier_line = LaplaceFourierBasis(L_line)
W_line = JointLaplacelet(L_line,
                         *log_wavelet_kernels_gen(3, 4, np.log(np.max(w_line))))


option = "One-shot-diffusion"#"One-shot-diffusion"
F_sol,F_irr = get_frequency_mask(B1,B2) # Get frequency bands
diff_order_sol= 1
diff_order_irr = 1
step_prog = 1
source_sol = np.ones((N1,))
source_irr = np.ones((N1,))
S_neigh, complete_coverage = cluster_on_neigh(B1,B2,diff_order_sol,diff_order_irr,source_sol,source_irr,option,step_prog)
R = [F_sol, F_irr]
S = S_neigh
top_K_coll = [2,4, None]
errs = np.logspace(-6, 0, 10, base=10)
complete_coll_sparsity = {}
min_ = 1e5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# orthogonal matching pursuit #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
for top_K_slep in top_K_coll:
    print("Complete Coverage: "+str(complete_coverage))
    W_slepians = SimplicianSlepians(B1, B2, S, R, top_K = top_K_slep, save_dir = res_dir, load_dir = res_dir)
    print("Slepians Rank: "+str(W_slepians.full_rank))
    print("Dictionary Dimension: "+str(W_slepians.atoms_flat.shape))
    slepians_sparsity = [len(W_slepians.omp(flow, err=err).coef_.nonzero()[0])
                            for err in errs]
    complete_coll_sparsity["slepians "+str(top_K_slep)]=slepians_sparsity

fourier_sparsity = [len(W_fourier.omp(flow, err=err).coef_.nonzero()[0])
                    for err in errs]
joint_sparsity = [len(W_joint.omp(flow, err=err).coef_.nonzero()[0])
                  for err in errs]
sep_sparsity = [len(W_sep.omp(flow, err=err).coef_.nonzero()[0])
                for err in errs]
lift_sparsity = [len(W_lift.omp(flow, err=err).coef_.nonzero()[0])
                 for err in errs]
lift_mixed_sparsity = [len(W_lift.omp(flow, err=err).coef_.nonzero()[0])
                       for err in errs]

fourier_line_sparsity = [len(W_fourier_line.omp(flow, err=err).coef_.nonzero()[0])
                         for err in errs]
line_sparsity = [len(W_line.omp(flow, err=err).coef_.nonzero()[0])
                 for err in errs]

sparsity_data = {
    'fourier':fourier_sparsity,
    'joint':joint_sparsity,
    'lift':lift_sparsity,
    'lift_mixed':lift_mixed_sparsity,
    'fourier_line':fourier_line_sparsity,
    'line':line_sparsity,
    'sep':sep_sparsity,
}

sparsity_data = {**sparsity_data,**complete_coll_sparsity}

sparsity_df = pd.DataFrame(sparsity_data,
                           columns=list(sparsity_data.keys()),
                           index=errs)
print(sparsity_df)
sparsity_df.to_csv(f'{res_dir}/sparsity.csv', float_format='%0.4f', index_label='err', sep = ";")

