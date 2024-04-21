from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings


def plot_error_curves(
    min_error_fou_test: np.ndarray,
    min_error_edge_test: np.ndarray,
    min_error_joint_test: np.ndarray,
    min_error_sep_test: np.ndarray,
    k0_coll: np.ndarray,
    dictionary_type: str) -> plt.Axes:
    """
    Plot the test errors curves for learning algorithms comparing Fourier, Edge, Joint, and Separated dictionary parametrization.

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


def plot_changepoints_curve(history, k0, nu, T, burn_in: float = 0):

    warnings.filterwarnings('ignore')
    start_iter = 0
    end_iter = 0
    change_points = []
    change_points_y1 = []
    change_points_y2 = []
    burn_in_iter = 0
    his=[]
    xx = []
    for i,h in enumerate(history):
        if i == 0:
            burn_in_iter=int(np.ceil(burn_in*len(h)))
        his+=h
        end_iter += len(h)-1
        tmp = range(start_iter, end_iter+1)
        xx += tmp
        start_iter = end_iter
        change_points.append(end_iter)
        change_points_y1.append(h[-1])
        change_points_y2.append(h[0])

    plt_data = pd.DataFrame({'y':his[burn_in_iter:], 
                    'x':xx[burn_in_iter:]})

    change_points = np.array(change_points[:-1])
    change_points_y1 = np.array(change_points_y1[:-1])
    change_points_y2 = np.array(change_points_y2[1:])

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})
    my_plt = sns.lineplot(x=plt_data['x'],y=plt_data['y'], estimator=None, sort=False)

    # Change-points
    sns.scatterplot(x=np.hstack([change_points, change_points]),
                    y=np.hstack([change_points_y1, change_points_y2]),
                    label='Change Points: optimally removing \n a triangle from Upper Laplacian.',
                    color='purple', marker='d')

    plt.vlines(x=change_points, color='lightblue', linestyle='dotted', 
            ymax=change_points_y1, ymin=change_points_y2)

    # Burn-in area
    plt.axvspan(0, burn_in_iter, color='grey', alpha=0.2, hatch='//')
    x0, xmax = plt.xlim()
    y0, ymax = plt.ylim()
    my_plt.set_title(f'Optimistic topology learning',fontsize=16, pad=25)
    plt.suptitle(f'Assumed signal sparsity: {k0}', fontsize=12, color='gray', x=0.5, y=0.92)
    plt.text(y=ymax*0.7, x=xmax*0.1, s=f'Burn-in: {burn_in_iter} iters.', fontsize=9, color='gray')
    plt.text(s=f' Number of inferred triangles: {nu - change_points.shape[0]} \n Number of true triangles: {nu-T}',
            y=ymax*0.9, x=xmax*0.75, fontsize=12, color='purple')
    my_plt.set_xlabel('Iteration')
    my_plt.set_ylabel('Error (log scale)')
    plt.xticks(change_points)
    plt.yticks([])
    plt.xlim(left=0)
    plt.yscale('log')
    plt.show() 

    return my_plt