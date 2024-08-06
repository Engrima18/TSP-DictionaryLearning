from tqdm import tqdm
import numpy as np
from .TspSolver import TspSolver
from .utils import final_save


@final_save
def param_dict_learning(
    X_train,
    X_test,
    Y_train,
    Y_test,
    c_true,
    epsilon_true,
    n_sim,
    topo_params,
    algo_params,
    K0_coll,
    Lu_true,
    prob_T,
    learn_topology=False,
    verbose: bool = True,
):
    """
    Learn the sparse representation and the dictionary atoms with the alternating-direction algorithm,
    for given test and training set, and comparing the performances in terms of training and test
    approximation error for several algorithmic setup, principally differencing by dictionary parameterization:
        - Fourier dictionary parameterization
        - Edge Laplacian dictionary parameterization
        - Joint Hodge Laplacian dictionary parameterization
        - Separated Hodge Laplacian dictionary parameterization
        - (Optionally) Separated Hodge Laplacian dictionary parameterization plus Topology Learning
    """

    min_error_fou_train = np.zeros((n_sim, len(K0_coll)))
    min_error_fou_test = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_train = np.zeros((n_sim, len(K0_coll)))
    min_error_sep_test = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_train = np.zeros((n_sim, len(K0_coll)))
    min_error_edge_test = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_train = np.zeros((n_sim, len(K0_coll)))
    min_error_joint_test = np.zeros((n_sim, len(K0_coll)))
    approx_fou = np.zeros((n_sim, len(K0_coll)))
    approx_sep = np.zeros((n_sim, len(K0_coll)))
    approx_joint = np.zeros((n_sim, len(K0_coll)))
    approx_edge = np.zeros((n_sim, len(K0_coll)))
    approx_comp = np.zeros((n_sim, len(K0_coll)))
    laplacian_fou = np.zeros((n_sim, len(K0_coll)))
    laplacian_sep = np.zeros((n_sim, len(K0_coll)))
    laplacian_joint = np.zeros((n_sim, len(K0_coll)))
    laplacian_edge = np.zeros((n_sim, len(K0_coll)))
    laplacian_comp = np.zeros((n_sim, len(K0_coll)))

    dict_errors = {
        "fou": (min_error_fou_train, min_error_fou_test, approx_fou, laplacian_fou),
        "edge": (
            min_error_edge_train,
            min_error_edge_test,
            approx_edge,
            laplacian_edge,
        ),
        "joint": (
            min_error_joint_train,
            min_error_joint_test,
            approx_joint,
            laplacian_joint,
        ),
        "sep": (min_error_sep_train, min_error_sep_test, approx_sep, laplacian_sep),
    }

    dict_types = {
        "fou": ("Fourier", "fourier"),
        "edge": ("Edge Laplacian", "edge_laplacian"),
        "joint": ("Hodge Laplacian", "joint"),
        "sep": ("Separated Hodge Laplacian", "separated"),
    }

    if learn_topology:
        min_error_complete_train = np.zeros((n_sim, len(K0_coll)))
        min_error_complete_test = np.zeros((n_sim, len(K0_coll)))
        dict_errors["comp"] = (
            min_error_complete_train,
            min_error_complete_test,
            approx_comp,
            laplacian_comp,
        )
        dict_types["comp"] = (
            "Separated Hodge Laplacian with Topology learning",
            "separated",
        )

    models = {}

    for sim in range(n_sim):

        for k0_index, k0 in tqdm(enumerate(K0_coll)):

            for d in dict_types.items():

                model = TspSolver(
                    X_train=X_train[:, :, sim],
                    X_test=X_test[:, :, sim],
                    Y_train=Y_train[:, :, sim],
                    Y_test=Y_test[:, :, sim],
                    c=c_true[sim],
                    epsilon=epsilon_true[sim],
                    K0=k0,
                    dictionary_type=d[1][1],
                    **topo_params,
                )

                try:

                    (
                        dict_errors[d[0]][0][sim, k0_index],
                        dict_errors[d[0]][1][sim, k0_index],
                        dict_errors[d[0]][2][sim, k0_index],
                    ) = model.fit(d[0], Lu_true, **algo_params)

                    models[f"{sim},{k0_index}"] = model

                    if verbose:
                        print(
                            f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Done! Test Error: {dict_errors[d[0]][1][sim,k0_index]:.3f}"
                        )
                        print(
                            f"Topology Approx. Error: {dict_errors[d[0]][2][sim,k0_index]:.3f}"
                        )
                except:
                    print(
                        f"Simulation: {sim+1}/{n_sim} Sparsity: {k0} Testing {d[1][0]}... Diverged!"
                    )
                    try:
                        (
                            dict_errors[d[0]][0][sim, k0_index],
                            dict_errors[d[0]][1][sim, k0_index],
                        ) = (
                            dict_errors[d[0]][0][sim - 1, k0_index],
                            dict_errors[d[0]][1][sim - 1, k0_index],
                        )
                    except:
                        (
                            dict_errors[d[0]][0][sim, k0_index],
                            dict_errors[d[0]][1][sim, k0_index],
                        ) = (
                            dict_errors[d[0]][0][sim + 1, k0_index],
                            dict_errors[d[0]][1][sim + 1, k0_index],
                        )

    return dict_errors, models


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

                model = TspSolver(
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

                model = TspSolver(
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
