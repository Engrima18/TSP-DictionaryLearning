import os
import pickle
import re
import pandas as pd
import numpy as np


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
