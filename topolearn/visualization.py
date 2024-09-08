from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from topolearn.utils import save_plot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@save_plot
def plot_error_curves(
    dict_errors,
    K0_coll,
    **kwargs,
) -> plt.Axes:
    """
    Plot the test error curves for learning algorithms comparing Fourier, Edge, Joint, and Separated dictionary parametrization.

    Parameters:
    - k0_coll (List[int]): Collection of sparsity levels.
    - dictionary_type (str): Type of dictionary used ('fou', 'edge', 'joint', 'sep', 'comp').

    Returns:
    - plt.Axes: The Axes object of the plot.
    """

    params = {"dictionary_type": "separated", "prob_T": 1.0, "test_error": True}

    params.update(kwargs)
    dictionary_type = params["dictionary_type"]
    prob_T = params["prob_T"]
    test_error = params["test_error"]

    dict_types = {
        "fourier": "Fourier",
        "edge": "Edge Laplacian",
        "joint": "Hodge Laplacian",
        "separated": "Separated Hodge Laplacian",
        "complete": "Separated Hodge Laplacian with topology learning",
    }
    TITLE = [dict_types[typ] for typ in dict_types.keys() if typ in dictionary_type][0]
    i = 0 if test_error else 1
    res_df = pd.DataFrame()
    for typ in dict_errors.keys():
        tmp_df = pd.DataFrame(dict_errors[typ][i])
        tmp_df.columns = K0_coll
        tmp_df = tmp_df.melt(var_name="Sparsity", value_name="Error")
        tmp_df["Method"] = dict_types[typ]
        res_df = pd.concat([res_df, tmp_df]).reset_index(drop=True)

    markers = (
        [">", "^", "v", "d"]
        if len(dict_errors.keys()) == 4
        else [">", "^", "v", "d", "s"]
    )
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Method",
        palette=sns.color_palette()[: len(dict_errors)],
        markers=markers,
        dashes=False,
        style="Method",
    )

    my_plt.set(yscale="log")
    my_plt.set_title(f"True dictionary: {TITLE}")
    xlabel = "Test" if test_error else "Training"
    my_plt.set_ylabel(f"{xlabel} NMSE (log scale)")

    return my_plt


@save_plot
def plot_topology_approx_errors(res_df, **kwargs):

    res_df = res_df.reset_index()
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Number of Triangles",
        palette=sns.color_palette("viridis", as_cmap=True),
    )

    my_plt.set(ylabel=r"$||L_u - \hat{L}_u^*||^2$")

    return my_plt


@save_plot
def plot_changepoints_curve(
    history,
    k0,
    nu,
    p,
    mode: str = "optimistic",
    burn_in: float = 0,
    a=0.1,
    b=0.1,
    c=0.7,
    d=0.7,
    yscale: str = "log",
    sparse_plot=False,
    include_burn_in=False,
    step_h=1.0,
    step_x=1.0,
    **kwargs,
):

    T = int(np.ceil(nu * (1 - p)))
    start_iter = 0
    end_iter = 0
    change_points = []
    change_points_y1 = []
    change_points_y2 = []
    burn_in_iter = 0
    his = []
    xx = []
    for i, h in enumerate(history):
        if i == 0:
            burn_in_iter = int(np.ceil(burn_in * len(h)))
        his += h
        end_iter += len(h) - 1
        tmp = range(start_iter, end_iter + 1)
        xx += tmp
        start_iter = end_iter
        change_points.append(end_iter)
        change_points_y1.append(h[-1])
        change_points_y2.append(h[0])

    plt_data = pd.DataFrame({"y": his[burn_in_iter:], "x": xx[burn_in_iter:]})

    change_points = np.array(change_points[:-1])
    change_points_y1 = np.array(change_points_y1[:-1])
    change_points_y2 = np.array(change_points_y2[1:])
    # change_points_y = plt_data[plt_data['x'].isin(change_points)].y.to_numpy()[np.arange(0, len(change_points), 1)]

    plt.figure(figsize=(14, 6))
    sns.set_style("whitegrid", {"grid.color": ".6", "grid.linestyle": ":"})

    my_plt = sns.lineplot(x=plt_data["x"], y=plt_data["y"], estimator=None, sort=False)

    labels = (
        ("removing", "Optimistic")
        if mode == "optimistic"
        else ("adding", "Pessimistic")
    )
    # Change-points
    sns.scatterplot(
        x=np.hstack([change_points, change_points]),
        y=np.hstack([change_points_y1, change_points_y2]),
        label=f"Change Points: optimally {labels[0]} \n a triangle from Upper Laplacian.",
        color="purple",
        marker="d",
    )

    plt.vlines(
        x=change_points,
        color="lightblue",
        linestyle="dotted",
        ymax=change_points_y1,
        ymin=change_points_y2,
    )

    # Burn-in area
    plt.axvspan(0, burn_in_iter, color="grey", alpha=0.2, hatch="//")

    if include_burn_in:
        x0, xmax = plt.xlim()
    else:
        x0, xmax = plt.xlim()
        x0 = burn_in_iter

    y0, ymax = plt.ylim()

    my_plt.set_title(f"{labels[1]} topology learning", fontsize=16, pad=25)
    plt.suptitle(
        f"Assumed signal sparsity: {k0}  -  Step size h: {step_h}  -  Step size X: {step_x}",
        fontsize=12,
        color="gray",
        x=0.5,
        y=0.92,
    )
    plt.text(
        y=ymax * a,
        x=xmax * b,
        s=f"Burn-in: {burn_in_iter} iters.",
        fontsize=15,
        color="gray",
    )
    plt.text(
        s=f" Number of inferred triangles: {nu - change_points.shape[0]} \n Number of true triangles: {nu-T}",
        y=ymax * c,
        x=xmax * d,
        fontsize=12,
        color="purple",
    )
    my_plt.set_xlabel("Iteration")

    if sparse_plot:
        tmp_vector = np.ones(len(change_points))
        tmp_vector[1::2] = 0
        plt.xticks(change_points * tmp_vector)
    else:
        if change_points.shape[0] > 1:
            plt.xticks(change_points)
    plt.xlim(left=x0, right=xmax)

    if yscale == "log":
        my_plt.set_ylabel("Log-Error")
        plt.yscale("log")
    else:
        my_plt.set_ylabel("Error")

    plt.yticks([])

    # Identify the region where y-values change slowly
    y_diff = np.abs(np.diff(plt_data["y"]))
    slow_change_indices = np.where(y_diff < 1e-2)[0]

    if len(slow_change_indices) > 0:
        # Select the first significant region of slow change
        zoom_start = slow_change_indices[0]
        zoom_end = slow_change_indices[-1]

        # Create inset axes for zoomed-in region
        ax_inset = inset_axes(
            my_plt, width="30%", height="40%", loc="upper right", borderpad=2
        )
        sns.lineplot(
            x=plt_data["x"].iloc[zoom_start:zoom_end],
            y=plt_data["y"].iloc[zoom_start:zoom_end],
            estimator=None,
            sort=False,
            ax=ax_inset,
        )

        # Set limits for the zoomed-in region
        ax_inset.set_xlim(plt_data["x"].iloc[zoom_start], plt_data["x"].iloc[zoom_end])
        ax_inset.set_ylim(
            plt_data["y"].iloc[zoom_start:zoom_end].min(),
            plt_data["y"].iloc[zoom_start:zoom_end].max(),
        )

        if yscale == "log":
            ax_inset.set_yscale("log")

        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title("Zoomed-in region", fontsize=10)
    plt.show()


@save_plot
def plot_learnt_topology(
    G_true,
    Lu_true,
    B2_true,
    model_gt,
    model_opt,
    model_pess=None,
    sub_size=100,
    **kwargs,
):

    try:
        topos = [model_gt, model_opt, model_pess]
        num_triangles = [
            model_gt.get_numb_triangles(),
            model_opt.get_numb_triangles("optimistic"),
            model_pess.get_numb_triangles("pessimistic"),
        ]
        incidence_mat = [B2_true, model_opt.B2, model_pess.B2]
        titles = [
            "True number of triangles: ",
            "Inferred number of triangles (optimistic method): ",
            "Inferred number of triangles (pessimistic method): ",
        ]
        _, axs = plt.subplots(1, 3, figsize=(16, 6))
    except:
        topos = [model_gt, model_opt]
        num_triangles = [
            model_gt.get_numb_triangles(),
            model_opt.get_numb_triangles("optimistic"),
        ]
        incidence_mat = [B2_true, model_opt.B2]
        titles = [
            "True number of triangles: ",
            "Inferred number of triangles (optimistic method): ",
        ]
        _, axs = plt.subplots(1, 2, figsize=(12, 6))

    i = 0

    for ax, title in zip(axs, titles):
        A = G_true.get_adjacency()
        tmp_G = nx.from_numpy_array(A)
        pos = nx.kamada_kawai_layout(tmp_G)
        nx.draw(tmp_G, pos, with_labels=False, node_color="purple", node_size=15, ax=ax)
        # num_triangles = 0

        for triangle_index in range(B2_true.shape[1]):
            np.random.seed(triangle_index)
            color = np.random.rand(3)
            triangle_vertices = []

            for edge_index, edge in enumerate(tmp_G.edges):
                if edge_index < sub_size:
                    if incidence_mat[i][edge_index, triangle_index] != 0:
                        pos1 = tuple(pos[edge[0]])
                        pos2 = tuple(pos[edge[1]])
                        if pos1 not in triangle_vertices:
                            triangle_vertices.append(pos1)
                        if pos2 not in triangle_vertices:
                            triangle_vertices.append(pos2)
            if triangle_vertices != []:
                # num_triangles += 1
                triangle_patch = Polygon(
                    triangle_vertices,
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    alpha=0.3,
                )
                ax.add_patch(triangle_patch)

        ax.set_title(title + str(num_triangles[i]))
        ax.text(
            0.5,
            -0,
            r"$||L_u - \hat{L}_u^*||^2$:"
            + f" {topos[i].get_topology_approx_error(Lu_true, 4)}         NMSE: {topos[i].get_test_error(4)}",
            ha="center",
            transform=ax.transAxes,
        )

        i += 1

    plt.tight_layout()


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
    n_sim = errors["qp"].shape[0]

    for algorithm, error_matrix in errors.items():
        for sim in range(n_sim):
            tmp_df = pd.DataFrame()
            tmp_df["Error"] = error_matrix[sim, :]
            tmp_df["Sparsity"] = k0_coll
            tmp_df["Algorithm"] = dict_types[algorithm]
            res_df = pd.concat([res_df, tmp_df])

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    my_plt = sns.lineplot(
        data=res_df,
        x="Sparsity",
        y="Error",
        hue="Algorithm",
        palette=sns.color_palette("husl"),
        markers=[">", "^", "v"],
        dashes=False,
        style="Algorithm",
    )
    my_plt.set(yscale="log")
    my_plt.set_title("Topology learning: algorithms comparison")
    my_plt.set_ylabel("Error (log scale)")

    return my_plt
