import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_error_curves(min_error_fou_test,
                      min_error_edge_test,
                      min_error_joint_test,
                      min_error_sep_test,
                      n_sim,
                      K0_coll):
    """Plots the error curves for Fourier, Edge and Joint tests"""
    
    TITLE = "Edge Laplacian"

    res_df = pd.DataFrame()
    dict_types = {"fou": "Fourier", "edge": "Edge Laplacian", "joint": "Hodge Laplacian", "sep": "Separated Hodge Laplacian"}

    for d in dict_types.items():
        for sim in range(n_sim):
            tmp_df = pd.DataFrame()   
            tmp_df["Error"] = eval(f'min_error_{d[0]}_test[sim,:]')
            tmp_df["Sparsity"] = K0_coll
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