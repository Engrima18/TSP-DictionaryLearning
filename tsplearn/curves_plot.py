from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_error_curves(
    min_error_fou_test: np.ndarray,
    min_error_edge_test: np.ndarray,
    min_error_joint_test: np.ndarray,
    min_error_sep_test: np.ndarray,
    k0_coll: np.ndarray,
    dictionary_type: str
) -> plt.Axes:
    """
    Plot the error curves for Fourier, Edge, Joint, and Separated test errors.

    Parameters:
    - min_error_fou_test (pd.DataFrame): DataFrame containing minimum errors for the Fourier test.
    - min_error_edge_test (pd.DataFrame): DataFrame containing minimum errors for the Edge test.
    - min_error_joint_test (pd.DataFrame): DataFrame containing minimum errors for the Joint test.
    - min_error_sep_test (pd.DataFrame): DataFrame containing minimum errors for the Separated test.
    - k0_coll (List[int]): Collection of sparsity levels.
    - dictionary_type (str): Type of dictionary used ('fou', 'edge', 'joint', 'sep').

    Returns:
    - plt.Axes: The Axes object of the plot.
    """

    dict_types = {"fou": "Fourier", "edge": "Edge Laplacian", "joint": "Hodge Laplacian", "sep": "Separated Hodge Laplacian"}
    TITLE = [dict_types[typ] for typ in dict_types.keys() if typ in dictionary_type][0]

    res_df = pd.DataFrame()
    n_sim = min_error_fou_test.shape[0]

    for d in dict_types.items():
        for _ in range(n_sim):
            tmp_df = pd.DataFrame()   
            tmp_df["Error"] = eval(f'min_error_{d[0]}_test[sim,:]')
            tmp_df["Sparsity"] = k0_coll
            tmp_df["Method"] = d[1]
            res_df = pd.concat([res_df, tmp_df])

    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    my_plt = sns.lineplot(data=res_df, x='Sparsity', y='Error', hue='Method',
                        palette=sns.color_palette(),
                        markers=['>', '^', 'v', 'd'], dashes=False, style='Method')
    my_plt.set(yscale='log')
    my_plt.set_title(f'True dictionary: {TITLE}')
    my_plt.set_ylabel('Error (log scale)')
    plt.show()

    return my_plt