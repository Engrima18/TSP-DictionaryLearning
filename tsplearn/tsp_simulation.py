



min_error_fou_train = np.zeros((n_sim, len(K0_coll)))
min_error_fou_test = np.zeros((n_sim, len(K0_coll)))
min_error_sep_train = np.zeros((n_sim, len(K0_coll)))
min_error_sep_test = np.zeros((n_sim, len(K0_coll)))
min_error_edge_train = np.zeros((n_sim, len(K0_coll)))
min_error_edge_test = np.zeros((n_sim, len(K0_coll)))
min_error_joint_train = np.zeros((n_sim, len(K0_coll)))
min_error_joint_test = np.zeros((n_sim, len(K0_coll)))

dict_errors = {
    "fou": (min_error_fou_train,min_error_fou_test),
    "edge": (min_error_edge_train,min_error_edge_test),
    "joint": (min_error_joint_train,min_error_joint_test),
    "sep": (min_error_sep_train,min_error_sep_test)
    }


dict_types = {
    "fou": ("Fourier","fourier"),
    "edge": ("Edge Laplacian", "edge_laplacian"),
    "joint": ("Hodge Laplacian","joint"),
    "sep": ("Separated Hodge Laplacian","separated")
    }

for sim in range(n_sim):
    c = c_true[sim]  
    epsilon = epsilon_true[sim] 
    for k0_index, k0 in tqdm(enumerate(K0_coll)):
        discard = 1
        while discard == 1:
            
            try:
                D0, X0, discard = initialize_dic(Lu, Ld, s, k, Y_train[:, :, sim], k0, dictionary_type, c, epsilon, "only_X")
            except:
                print("Initialization Failed!")

        for d in dict_types.items():
            try:
                dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index], _, _, _ = topological_dictionary_learn(Y_train[:,:,sim], Y_test[:,:,sim],
                                                                                                                            k, n, s, D0, X0, Lu, Ld, d[1][1],
                                                                                                                            c, epsilon, k0, lambda_, max_iter,
                                                                                                                            patience, tol)
                if verbose:
                    print(f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Done! Test Error: {dict_errors[d[0]][1][sim,k0_index]}")
            except:
                print(f'Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Diverged!')
                try:
                    dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index] = (dict_errors[d[0]][0][sim-1,k0_index]
                                                                                          , dict_errors[d[0]][1][sim-1,k0_index])
                except:
                    dict_errors[d[0]][0][sim,k0_index], dict_errors[d[0]][1][sim,k0_index] = (dict_errors[d[0]][0][sim+1,k0_index]
                                                                                          , dict_errors[d[0]][1][sim+1,k0_index])
