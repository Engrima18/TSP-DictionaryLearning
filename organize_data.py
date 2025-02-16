import pickle
import os
import numpy as np
import pandas as pd
import re
import glob

print


def get_pickle_files(directory):
    pattern = os.path.join(directory, "tri_learn*.pkl")
    return glob.glob(pattern)


def extract_info(file):
    s_pre = file.split(".")[0]
    # matches = re.findall(r"(?<=[A-Za-z])((?:\d{2})+)", s_pre)
    # tr_list = []
    # for match in matches:
    #     tr_list.extend([int(match[i : i + 2]) for i in range(0, len(match), 2)])
    # word_match = re.search(r"(greedy|soft)", s_pre)
    # algo_method = word_match.group(1) if word_match else None

    match = re.search(r"(greedy|soft)((?:\d{2})+)", s_pre)
    if match:
        algo_method = match.group(1)
        digits_str = match.group(2)
        # Split the digit string into 2-digit integers.
        tr_list = [int(digits_str[i : i + 2]) for i in range(0, len(digits_str), 2)]
    else:
        algo_method = None
        tr_list = []

    with open(file, "rb") as f:
        models = pickle.load(f)
        dict_errors = pickle.load(f)
    return tr_list, algo_method, models, dict_errors


def organize_data(sparsity_mode="max"):
    path = os.getcwd()
    df = pd.DataFrame()
    df_mse = pd.DataFrame()
    data_sparsity = [5, 15]  # [5, 15, 25]
    for s in data_sparsity:
        directory = f"{path}\\results\\final\\{sparsity_mode}_sparsity{s}"
        files = get_pickle_files(directory)
        for file in files:
            print(file)
            tr_list, algo_method, models, dict_errors = extract_info(file)
            # print(tr_list)
            for i, p in enumerate(tr_list):
                for sim in range(10):
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                {
                                    "Error": dict_errors[i][f"complete_{algo_method}"][
                                        2
                                    ][sim],
                                    "Method": algo_method,
                                    "Sim": sim,
                                    "Sparsity": s,
                                    "Triangles": p,
                                }
                            ),
                        ]
                    )
                    a = []
                    for arr in models[i][(sim, 0)][0].train_error_hist:
                        a += arr
                    df_mse = pd.concat(
                        [
                            df_mse,
                            pd.DataFrame(
                                {
                                    "MSE": a,
                                    "Method": algo_method,
                                    "Sim": sim,
                                    "Sparsity": s,
                                    "Triangles": p,
                                }
                            ),
                        ]
                    )

    with open(f"{path}\\results\\paper\\lu_error2.pkl", "wb") as ff:
        pickle.dump(df, ff)
        pickle.dump(df_mse, ff)


if __name__ == "__main__":
    organize_data()
