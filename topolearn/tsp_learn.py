from tqdm import tqdm
import numpy as np
from .TopoSolver import TopoSolver


# SDP vs QP vs QP complete... comment the function
def complete_learning_test(
    X_train,
    X_test,
    Y_train,
    Y_test,
    c_true,
    epsilon_true,
    n_sim,
    topo_params,
    K0_coll,
    max_iter,
    patience,
    tol,
    lambda_,
    step_h: int = 1,
    step_x: int = 1,
    verbose: bool = True,
    include_sdp: bool = False,
):

    min_error_qp_test = np.zeros((n_sim, len(K0_coll)))
    min_error_qp_comp_test = np.zeros((n_sim, len(K0_coll)))

    algo_errors = {
        "qp": min_error_qp_test,
        "qp_comp": min_error_qp_comp_test,
    }

    algo_types = {"qp": ("QP", True, False), "qp_comp": ("QP complete", True, True)}

    if include_sdp:
        min_error_sdp_test = np.zeros((n_sim, len(K0_coll)))
        algo_errors["sdp"] = min_error_sdp_test
        algo_types["sdp"] = ("SDP", False, False)

    for sim in range(n_sim):

        for k0_index, k0 in tqdm(enumerate(K0_coll)):

            for a in algo_types.items():

                model = TopoSolver(
                    X_train=X_train[:, :, sim],
                    X_test=X_test[:, :, sim],
                    Y_train=Y_train[:, :, sim],
                    Y_test=Y_test[:, :, sim],
                    c=c_true[sim],
                    epsilon=epsilon_true[sim],
                    K0=k0,
                    dictionary_type="separated",
                    **topo_params,
                )

                try:
                    # Complete learning
                    if a[1][2]:
                        algo_errors[a[0]][sim, k0_index], _, _, _ = (
                            model.learn_upper_laplacian(
                                lambda_=lambda_,
                                max_iter=max_iter,
                                patience=patience,
                                tol=tol,
                                verbose=False,
                                step_h=step_h,
                                step_x=step_x,
                                QP=a[1][1],
                            )
                        )
                    # Learn only the dictionary (no topology)
                    else:
                        if a[1][1]:
                            algo_errors[a[0]][sim, k0_index], _, _ = (
                                model.topological_dictionary_learn_qp(
                                    lambda_=lambda_,
                                    max_iter=max_iter,
                                    patience=patience,
                                    tol=tol,
                                    step_h=step_h,
                                    step_x=step_x,
                                )
                            )

                        else:
                            algo_errors[a[0]][sim, k0_index], _, _ = (
                                model.topological_dictionary_learn(
                                    lambda_=lambda_,
                                    max_iter=max_iter,
                                    patience=patience,
                                    tol=tol,
                                    step_h=step_h,
                                    step_x=step_x,
                                )
                            )

                    if verbose:
                        print(
                            f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} learning with {a[1][0]}... Done! Test Error: {algo_errors[a[0]][sim,k0_index]}"
                        )
                except:
                    print(
                        f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {a[1][0]}... Diverged!"
                    )
                    # If diverged, simply interpolate
                    try:
                        algo_errors[a[0]][sim, k0_index] = algo_errors[a[0]][
                            sim - 1, k0_index
                        ]
                    except:
                        algo_errors[a[0]][sim, k0_index] = algo_errors[a[0]][
                            sim - 1, k0_index
                        ]

    return algo_errors


# Slepians + Wavelet + Fourier + Edge Laplacian + Separated QP + Separated QP complete
def simulate_learnable_vs_analytic_dict(
    X_train,
    X_test,
    Y_train,
    Y_test,
    c_true,
    epsilon_true,
    n_sim,
    topo_params,
    K0_coll,
    max_iter,
    patience,
    tol,
    lambda_,
    step_h: int = 1,
    step_x: int = 1,
    verbose: bool = True,
):

    min_error_slep_test = np.zeros((n_sim, len(K0_coll)))
    min_error_wave_test = np.zeros((n_sim, len(K0_coll)))
    min_error_fou_test = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_test = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_test = np.zeros((n_sim, len(K0_coll)))
    min_error_comp_test = np.zeros((n_sim, len(K0_coll)))

    algo_errors = {
        "slep": min_error_slep_test,
        "wave": min_error_wave_test,
        "fou": min_error_fou_test,
        "edge": min_error_edge_test,
        "sep": min_error_sep_test,
        "comp": min_error_comp_test,
    }

    algo_types = {
        "slep": ("slepians", "Slepians", False),
        "wave": ("wavelet", "Wavelets", False),
        "fou": ("fourier", "Fourier", False),
        "edge": ("edge_laplacian", "Edge Laplacian", False),
        "sep": ("separated", "Separated Hodge Laplacian", False),
        "comp": ("separated", "Separated Hodge + Topology", True),
    }

    for sim in range(n_sim):

        for k0_index, k0 in tqdm(enumerate(K0_coll)):

            for a in algo_types.items():

                model = TopoSolver(
                    X_train=X_train[:, :, sim],
                    X_test=X_test[:, :, sim],
                    Y_train=Y_train[:, :, sim],
                    Y_test=Y_test[:, :, sim],
                    c=c_true[sim],
                    epsilon=epsilon_true[sim],
                    K0=k0,
                    dictionary_type=a[1][0],
                    **topo_params,
                )

                try:
                    if a[1][2]:
                        algo_errors[a[0]][sim, k0_index], _, _, _ = (
                            model.learn_upper_laplacian(
                                lambda_=lambda_,
                                max_iter=max_iter,
                                patience=patience,
                                tol=tol,
                                verbose=False,
                                step_h=step_h,
                                step_x=step_x,
                                QP=True,
                            )
                        )

                    else:
                        algo_errors[a[0]][sim, k0_index], _, _ = (
                            model.topological_dictionary_learn_qp(
                                lambda_=lambda_,
                                max_iter=max_iter,
                                patience=patience,
                                tol=tol,
                                step_h=step_h,
                                step_x=step_x,
                            )
                        )

                    if verbose:
                        print(
                            f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} learning with {a[1][1]}... Done! Test Error: {algo_errors[a[0]][sim,k0_index]}"
                        )

                except:
                    print(
                        f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {a[1][0]}... Diverged!"
                    )
                    # If diverged, simply interpolate
                    try:
                        algo_errors[a[0]][sim, k0_index] = algo_errors[a[0]][
                            sim - 1, k0_index
                        ]
                    except:
                        algo_errors[a[0]][sim, k0_index] = algo_errors[a[0]][
                            sim - 1, k0_index
                        ]

    return algo_errors
