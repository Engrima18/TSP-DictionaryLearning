import os
import pickle
import re
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import numpy as np
import warnings
from topolearn import (
    plot_learnt_topology,
    plot_algo_errors,
    plot_error_curves,
    plot_changepoints_curve,
    plot_analytic_error_curves,
    plot_topology_approx_errors,
    EnhancedGraph,
    TopoSolver,
)

warnings.filterwarnings("ignore")


def plot_main_curves(
    curves_params, res_path, K0_coll, analytic_res_path=None, analytic=False
):

    with open(res_path, "rb") as file:
        models = pickle.load(file)
        res = pickle.load(file)
    if analytic:
        with open(analytic_res_path, "rb") as file:
            _ = pickle.load(file)
            analyt_res = pickle.load(file)
        print("here")
        plot_analytic_error_curves(
            analytic_dict_errors=analyt_res,
            dict_errors=res,
            K0_coll=K0_coll,
            **curves_params,
        )
        return res, models
    # Plot test error curves
    plot_error_curves(dict_errors=res, K0_coll=K0_coll, **curves_params)
    # Plot training error curves
    curves_params["test_error"] = False
    plot_error_curves(dict_errors=res, K0_coll=K0_coll, **curves_params)

    return res, models


def fit_gt_model(i, p, G, Lu, K0_coll, data_path, cfg):

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    init_params = {
        "J": cfg.J,
        "P": cfg.P,
        "c": data["c_true"][cfg.n_sim - 1],
        "epsilon": data["epsilon_true"][cfg.n_sim - 1],
        "true_prob_T": p,
        "sub_size": cfg.sub_size,
        "seed": cfg.seed,
        "n": cfg.n,
        "K0": K0_coll[i],
        "p_edges": cfg.p_edges,
        "dictionary_type": "separated",
        "G_true": G,
    }

    algo_params = {
        "lambda_": cfg.lambda_,
        "tol": cfg.tol,
        "patience": cfg.patience,
        "max_iter": cfg.max_iter,
        "QP": True,
        "mode": "optimistic",
        "verbose": False,
    }

    gt_model = TopoSolver(
        X_train=data["X_train"][:, :, cfg.n_sim - 1],
        X_test=data["X_test"][:, :, cfg.n_sim - 1],
        Y_train=data["Y_train"][:, :, cfg.n_sim - 1],
        Y_test=data["Y_test"][:, :, cfg.n_sim - 1],
        **init_params,
    )

    gt_model.fit(
        Lu_true=Lu,
        init_mode="only_X",
        learn_topology=False,
        **algo_params,
    )

    return gt_model


@hydra.main(config_path="config", config_name="visualization", version_base="1.3")
def main(cfg: DictConfig):

    path = os.getcwd()
    K0_coll = np.arange(cfg.min_sparsity, cfg.max_sparsity, cfg.sparsity_freq)
    dict_types = ["separated", "edge", "joint"]
    p_triangles = cfg.p_triangles_list
    s_modes = cfg.sparsity_mode_list
    max_s = cfg.max_sparsity_list
    res_type = cfg.res_type

    for d in dict_types:
        for mode in s_modes:
            for s in max_s:
                dir_path = f"{path}\\results\\final\\{mode}_sparsity{s}"
                if d == "separated":
                    complete = True
                    res_df = pd.DataFrame()
                    for p in p_triangles:
                        curves_params = {
                            "dictionary_type": d,
                            "test_error": True,
                            "prob_T": p,
                            "sparsity_mode": mode,
                            "sparsity": s,
                        }
                        if res_type == "analytic":
                            res_path = f"{dir_path}\\res_{d}_T{int(p*100)}.pkl"
                            analytic_res_path = (
                                f"{dir_path}\\res_{d}_T{int(p*100)}_analyt.pkl"
                            )
                            try:
                                print(f"Here {analytic_res_path}")
                                res, models = plot_main_curves(
                                    curves_params,
                                    res_path,
                                    K0_coll,
                                    analytic_res_path,
                                    True,
                                )
                            except FileNotFoundError:
                                pass
                        else:
                            res_path = f"{dir_path}\\res_{d}_T{int(p*100)}.pkl"
                            pess_res_path = (
                                f"{dir_path}\\res_{d}_T{int(p*100)}_pess.pkl"
                            )
                            data_path = f"{path}\\synthetic_data\\{mode}_sparsity{s}\\top_data_T{int(p*100)}.pkl"
                            print(f"Prova: {res_path}")
                            try:
                                # Plot training and test sparse representation error curves
                                res, models = plot_main_curves(
                                    curves_params, res_path, K0_coll
                                )
                                with open(pess_res_path, "rb") as file:
                                    pess_models = pickle.load(file)
                                    _ = pickle.load(file)

                                if p != 1.0:

                                    for i, k in enumerate(K0_coll):
                                        curves_params["algo_sparsity"] = k
                                        example_model = models[f"{cfg.n_sim-1},{i}"]
                                        pess_example_model = pess_models[
                                            (cfg.n_sim - 1, i)
                                        ][0]
                                        G = EnhancedGraph(
                                            n=cfg.n,
                                            p_edges=cfg.p_edges,
                                            p_triangles=p,
                                            seed=cfg.seed,
                                        )
                                        Lu, _, _ = G.get_laplacians(
                                            sub_size=cfg.sub_size
                                        )
                                        B2 = G.get_b2()
                                        B2 = B2[: cfg.sub_size, :]
                                        B2 = B2[:, np.sum(np.abs(B2), 0) == 3]
                                        B2 = B2 @ G.mask
                                        gt_model = fit_gt_model(
                                            i, p, G, Lu, K0_coll, data_path, cfg
                                        )
                                        plot_learnt_topology(
                                            G,
                                            Lu,
                                            B2,
                                            gt_model,
                                            example_model,
                                            pess_example_model,
                                            cfg.sub_size,
                                            **curves_params,
                                        )

                                    tmp_df = pd.DataFrame(res["complete"][2])
                                    tmp_df.columns = K0_coll
                                    tmp_df = tmp_df.melt(
                                        var_name="Sparsity", value_name="Error"
                                    )

                                    tmp_df["Number of Triangles"] = cfg.nu - int(
                                        np.ceil(cfg.nu * (1 - p))
                                    )
                                    res_df = pd.concat([res_df, tmp_df])

                            except FileNotFoundError:
                                complete = False
                                print(f"No results found for {res_path}")
                        # if complete:
                        #     # Plot topology approximation error
                        #     plot_topology_approx_errors(res_df, **curves_params)

                else:
                    p = 1.0
                    curves_params = {
                        "dictionary_type": d,
                        "test_error": True,
                        "prob_T": p,
                        "sparsity_mode": mode,
                        "sparsity": s,
                    }
                    res_path = f"{dir_path}\\res_{d}_T{int(p*100)}.pkl"
                    if res_type == "analytic":
                        analytic_res_path = (
                            f"{dir_path}\\res_{d}_T{int(p*100)}_analyt.pkl"
                        )
                        try:
                            res, _ = plot_main_curves(
                                curves_params,
                                res_path,
                                K0_coll,
                                analytic_res_path,
                                True,
                            )
                        except FileNotFoundError:
                            pass
                    else:
                        try:
                            print(f"Prova: {res_path}")
                            res, _ = plot_main_curves(curves_params, res_path, K0_coll)
                        except FileNotFoundError:
                            print(f"No results found for {res_path}")


if __name__ == "__main__":
    main()
