from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import networkx as nx
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
        for sim in range(n_sim):
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


def plot_changepoints_curve(history, 
                            k0, 
                            nu, 
                            T, 
                            mode,
                            burn_in: float = 0, 
                            a=0.1, 
                            b=0.1, 
                            c=0.7, 
                            d=0.7,
                            sparse_plot=False,
                            include_burn_in=False,
                            step_h=1.,
                            step_x=1.):

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
    # change_points_y = plt_data[plt_data['x'].isin(change_points)].y.to_numpy()[np.arange(0, len(change_points), 1)]

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

    my_plt = sns.lineplot(x=plt_data['x'],y=plt_data['y'], estimator=None, sort=False)

    labels = ("removing", "Optimistic") if mode=="optimistic" else ("adding", "Pessimistic")
    # Change-points
    sns.scatterplot(x=np.hstack([change_points, change_points]),
                    y=np.hstack([change_points_y1, change_points_y2]),
                    label=f'Change Points: optimally {labels[0]} \n a triangle from Upper Laplacian.',
                    color='purple', marker='d')

    plt.vlines(x=change_points, color='lightblue', linestyle='dotted', 
            ymax=change_points_y1, ymin=change_points_y2)

    # Burn-in area
    plt.axvspan(0, burn_in_iter, color='grey', alpha=0.2, hatch='//')

    if include_burn_in:
        x0, xmax = plt.xlim()
    else:
        x0, xmax = plt.xlim()
        x0 = burn_in_iter

    y0, ymax = plt.ylim()

    my_plt.set_title(f'{labels[1]} topology learning',fontsize=16, pad=25)
    plt.suptitle(f'Assumed signal sparsity: {k0}  -  Step size h: {step_h}  -  Step size X: {step_x}', fontsize=12, color='gray', x=0.5, y=0.92)
    plt.text(y=ymax*a, x=xmax*b, s=f'Burn-in: {burn_in_iter} iters.', fontsize=15, color='gray')
    plt.text(s=f' Number of inferred triangles: {nu - change_points.shape[0]} \n Number of true triangles: {nu-T}',
            y=ymax*c, x=xmax*d, fontsize=12, color='purple')
    my_plt.set_xlabel('Iteration')
    my_plt.set_ylabel('Log-Error')
    
    if sparse_plot:
        tmp_vector = np.ones(len(change_points))
        tmp_vector[1::2] = 0
        plt.xticks(change_points*tmp_vector)
    else:
        if change_points.shape[0]>1:
            plt.xticks(change_points)
    plt.xlim(left=x0, right=xmax)
    plt.yscale('log')
    plt.yticks([])
    plt.show() 


def plot_learnt_topology(G_true, B2_true, topology1, topology2, sub_size):
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 6))
    incidence_mat = [B2_true, topology1.B2, topology2.B2]
    titles = ["True topology", "Inferred topology (pessimistic method)", "Inferred topology (optimistic method)"]
    i=0
    
    for ax, title in zip(axs, titles):
        A = G_true.get_adjacency()
        tmp_G = nx.from_numpy_array(A)
        pos = nx.kamada_kawai_layout(tmp_G)
        nx.draw(tmp_G, pos, with_labels=False, node_color='purple', node_size=15, ax=ax)
        num_triangles=0
        
        for triangle_index in range(B2_true.shape[1]):
            np.random.seed(triangle_index)
            color = np.random.rand(3)
            triangle_vertices=[]
            
            for edge_index, edge in enumerate(tmp_G.edges):
                if edge_index < sub_size:
                    if incidence_mat[i][edge_index, triangle_index] != 0:
                        pos1 = tuple(pos[edge[0]])
                        pos2 = tuple(pos[edge[1]])
                        if pos1 not in triangle_vertices:
                            triangle_vertices.append(pos1)
                        if pos2 not in triangle_vertices:
                            triangle_vertices.append(pos2)                
            if triangle_vertices!= []:
                num_triangles += 1
                triangle_patch = Polygon(triangle_vertices, closed=True, facecolor=color, edgecolor='black', alpha=0.3)
                ax.add_patch(triangle_patch)
        i+=1

        ax.set_title(title)
        ax.text(0.5, -0, "Number of triangles: {}".format(num_triangles), ha='center', transform=ax.transAxes)
    plt.tight_layout()
    plt.show()


def plot_algo_errors(errors: dict[str, np.ndarray], k0_coll: np.ndarray) -> plt.Axes:
    """
    Plot the algorithm errors against the sparsity levels, comparing different implementations
    of algorithms for learning representations of topological signals:
    - Semi-definite Programming dictionary and sparse representation joint learning (SDP);
    - Quadratic Programming dictionary and sparse representation joint learning (QP);
    - Quadratic Programming dictionary, sparse representation and topology (upper laplacian) joint learning (QP COMPLETE).

    All of the above methods in this case are considered in the "Separated Hodge" laplacian parametrization setup.

    Parameters:
    errors (dict[str, np.ndarray]): A dictionary containing error matrices for different algorithms.
                                    The keys are algorithm names, and the values are 2D numpy arrays
                                    where each row represents the errors for a single simulation.
    k0_coll (np.ndarray): An array of sparsity levels.

    Returns:
    plt.Axes: The axes object with the plotted data.
    """
    dict_types = {"qp": "QP", "sdp": "SDP", "qp_comp": "QP COMPLETE"}

    res_df = pd.DataFrame()
    n_sim = errors['qp'].shape[0]

    for algorithm, error_matrix in errors.items():
        for sim in range(n_sim):
            tmp_df = pd.DataFrame()
            tmp_df["Error"] = error_matrix[sim, :]
            tmp_df["Sparsity"] = k0_coll
            tmp_df["Algorithm"] = dict_types[algorithm]
            res_df = pd.concat([res_df, tmp_df])

    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    my_plt = sns.lineplot(data=res_df, x='Sparsity', y='Error', hue='Algorithm',
                          palette=sns.color_palette("husl"),
                          markers=['>', '^', 'v'], dashes=False, style='Algorithm')
    my_plt.set(yscale='log')
    my_plt.set_title('Topology learning: algorithms comparison')
    my_plt.set_ylabel('Error (log scale)')
    plt.show()

    return my_plt
