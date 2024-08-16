import os
import pickle
import re
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse
import pandas as pd
import numpy as np
from topolearn import (
    plot_learnt_topology,
    plot_algo_errors,
    plot_error_curves,
    plot_changepoints_curve,
)

parser = argparse.ArgumentParser(
    description="Run dictionary and topology learning with specified configuration."
)
parser.add_argument(
    "--config-dir", type=str, default="config", help="Configuration directory"
)
parser.add_argument(
    "--config-name",
    type=str,
    default="visualization.yaml",
    help="Configuration file name",
)
args = parser.parse_args()


def compare_Lu_approx(K0_coll, nu, res_dir="results\\final"):
    res_df = pd.DataFrame()
    PATH = os.path.join(os.getcwd(), res_dir)
    pattern = re.compile(r"(\d{2})\.(pkl|pickle)$")

    if not os.path.exists(PATH):
        print(f"The directory {PATH} does not exist.")
    else:
        for filename in os.listdir(PATH):
            # Check pickle files
            if filename.endswith(".pkl") or filename.endswith(".pickle"):

                file_path = os.path.join(PATH, filename)
                match = pattern.search(filename)
                if match:
                    triangles = match.group(1)

                try:
                    with open(file_path, "rb") as f:
                        _ = pickle.load(f)
                        res = pickle.load(f)

                    tmp_df = pd.DataFrame(res["complete"][2])
                    tmp_df.columns = K0_coll
                    tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
                    tmp_df["Number of Triangles"] = nu - int(
                        np.ceil(nu * (1 - float(triangles) / 100))
                    )
                    res_df = pd.concat([res_df, tmp_df])

                except Exception as e:
                    print(f"Failed to process {filename}: {e}")

    return res_df


@hydra.main(
    config_path=args.config_dir, config_name=args.config_name, version_base=None
)
def main(cfg: DictConfig):

    path = os.getcwd()
    K0_coll = np.arange(cfg.min_sparsity, cfg.max_sparsity, cfg.sparsity_freq)
    dict_types = ["separated", "edge", "joint"]
    p_triangles = cfg.p_triangles_list

    for d in dict_types:
        if d == "separated":
            for p in p_triangles:

                # data_path = f"{path}\\synthetic_data\\{name}.pkl"
                res_path = f"{path}\\results\\final\\resT{int(p*100)}.pkl"

                with open(res_path, "rb") as file:
                    models = pickle.load(file)
                    res = pickle.load(file)

                # Plot test error curves
                curves_params = {"dictionary_type": d, "test_error": True, "prob_T": p}
                plot_error_curves(dict_errors=res, K0_coll=K0_coll, **curves_params)
                # Plot training error curves
                curves_params["test_error"] = False
                plot_error_curves(dict_errors=res, K0_coll=K0_coll, **curves_params)
        else:
            pass


if __name__ == "__main__":
    main()
