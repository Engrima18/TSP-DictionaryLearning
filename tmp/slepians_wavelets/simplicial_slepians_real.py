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
import scipy.io
from sklearn.preprocessing import normalize 
from collections import defaultdict

res_dir = "/home/clabat/Dropbox/Topological_Slepians/Source/Results"

def compute_error(Y,X):
    num_col = Y.shape[1]
    error = 0
    for col in range(num_col):
        error = error + np.linalg.norm(Y[:,col]-X[:,col])**2/np.linalg.norm(Y[:,col])**2
    return (1/num_col)*error

#~~~~~~~~~~~~~~~#
# data loading  #
#~~~~~~~~~~~~~~~#
mat = scipy.io.loadmat('data_real.mat')
flow = mat["signal_edge"][:,~np.all(mat["signal_edge"] == 0, axis = 0)]
flow = flow/np.max(flow)
flow = flow[:,10:]
B1 = mat["B1"]
B2 = mat["B2"]
N0 = B1.shape[0]
N1 = B1.shape[1]
N2 = B2.shape[1]
#B1, B2 = scale_incidence_matrices(B1, B2)
#sgn_change = np.diag(np.sign(np.random.randn(N1)))
#B1 = B1@sgn_change
#B2 = sgn_change@B2
L, L1, L2 = hodge_laplacians(B1, B2)


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
spars_level = list(range(10,80,10))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# orthogonal matching pursuit #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
error_data = defaultdict(list)    
for top_K_slep in top_K_coll:
    print("Complete Coverage: "+str(complete_coverage))
    W_slepians = SimplicianSlepians(B1, B2, S, R, top_K = top_K_slep, save_dir = res_dir, load_dir = res_dir)
    print("Slepians Rank: "+str(W_slepians.full_rank))
    print("Dictionary Dimension: "+str(W_slepians.atoms_flat.shape))
    slepians_noisy = [W_slepians.omp(flow, k = spars)
                            for spars in spars_level]
    slepians_error = [compute_error(flow, W_slepians.atoms_flat@spars.coef_.T)
                            for spars in slepians_noisy]
    error_data["slepians "+str(top_K_slep)]= slepians_error

fourier_noisy = [W_fourier.omp(flow, k = spars)
                        for spars in spars_level]
fourier_error = [compute_error(flow, W_fourier.atoms_flat@spars.coef_.T)
                                for spars in fourier_noisy]

sep_noisy = [W_sep.omp(flow, k = spars)
                        for spars in spars_level]
sep_error = [compute_error(flow, W_sep.atoms_flat@spars.coef_.T)
                                for spars in sep_noisy]

error_data['fourier']=fourier_error
error_data['sep']=sep_error
"""   
error_data = {**error_data,**complete_coll_error}
for key in error_data.keys():
    error_data[key] = np.mean(np.array(error_data[key]),axis = 0)
"""
print(error_data)
error_df = pd.DataFrame(error_data,
                            columns=list(error_data.keys()),
                            index=spars_level)
print(error_df)
#error_df.to_csv(f'{res_dir}/error_snr_'+str(snr)+'.csv', float_format='%0.4f', index_label='err', sep = ";")

