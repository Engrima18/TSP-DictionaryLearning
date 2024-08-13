import pickle
import os
import dill
from topolearn import curves_plot


def show_and_save(
    dictionary_type,
    K0_coll,
    min_error_edge_test,
    min_error_edge_train,
    min_error_fou_test,
    min_error_fou_train,
    min_error_joint_test,
    min_error_joint_train,
    min_error_sep_test,
    min_error_sep_train,
):

    my_plt = curves_plot(
        min_error_fou_test,
        min_error_edge_test,
        min_error_joint_test,
        min_error_sep_test,
        K0_coll,
        dictionary_type,
    )

    save_var = {
        "min_error_edge_test": min_error_edge_test,
        "min_error_edge_train": min_error_edge_train,
        "min_error_fou_test": min_error_fou_test,
        "min_error_fou_train": min_error_fou_train,
        "min_error_joint_test": min_error_joint_test,
        "min_error_joint_train": min_error_joint_train,
        "min_error_sep_test": min_error_sep_test,
        "min_error_sep_train": min_error_sep_train,
    }

    PATH = os.getcwd()
    DIR_PATH = f"{PATH}\\results\\{dictionary_type}"
    FILENAME_ERR = f"{DIR_PATH}\\error.pkl"
    FILENAME_ENV = f"{DIR_PATH}\\ipynb_env.db"
    FILENAME_PLT = f"{DIR_PATH}\\plot.png"

    if not os.path.exists(DIR_PATH):
        os.makedirs(DIR_PATH)

    with open(FILENAME_ERR, "wb") as f:
        pickle.dump(save_var, f)
    f.close()

    dill.dump_session(FILENAME_ENV)

    fig = my_plt.get_figure()
    fig.savefig(FILENAME_PLT)
