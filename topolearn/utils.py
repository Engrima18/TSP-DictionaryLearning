import pickle
import os
import hashlib
from functools import wraps
import matplotlib.pyplot as plt


def memoize(file_name="cache\\DataTS.pkl"):
    """
    Decorator for caching function results in a pickle file
    and speed up the calculation.
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):

            graph_hash = hashlib.sha256(
                str((sorted(self.edges()), self.id)).encode()
            ).hexdigest()
            key = f"{func.__name__}_{graph_hash}"

            path = os.path.join(os.getcwd(), file_name)
            if os.path.exists(path):

                with open(path, "rb") as file:
                    try:
                        data = pickle.load(file)
                    except EOFError:
                        data = {}
                if key in data:
                    return data[key]

            result = func(self, *args, **kwargs)
            try:
                if os.path.exists(path):
                    with open(path, "rb") as file:
                        data = pickle.load(file)
                else:
                    data = {}
            except EOFError:
                data = {}
            data[key] = result
            with open(path, "wb") as file:
                pickle.dump(data, file)

            return result

        return wrapper

    return decorator


def memoize_or_save(func):
    @wraps(func)
    def wrapper(Lu, Ld, **kwargs):
        dictionary_type = kwargs.get("dictionary_type", "separated")
        prob_T = kwargs.get("prob_T", 1)
        sparsity_mode = kwargs.get("sparsity_mode", "max")

        if prob_T == 1:
            name = f"full_data_{dictionary_type}"
        else:
            name = f"top_data_T{int(prob_T*100)}"
        path = os.getcwd()
        dir = "max_sparsity" if sparsity_mode == "max" else "random_sparsity"
        dir_path = f"{path}\\synthetic_data\\{dir}"
        filename = f"{dir_path}\\{name}.pkl"

        try:
            # Try to load data from file
            with open(filename, "rb") as f:
                data = pickle.load(f)
            f.close()
        except FileNotFoundError:
            # If file not found, generate data and save
            D_true, Y_train, Y_test, X_train, X_test, epsilon_true, c_true = func(
                Lu, Ld, **kwargs
            )
            data = {
                "D_true": D_true,
                "Y_train": Y_train,
                "Y_test": Y_test,
                "X_train": X_train,
                "X_test": X_test,
                "epsilon_true": epsilon_true,
                "c_true": c_true,
            }

            os.makedirs(dir_path, exist_ok=True)
            with open(filename, "wb") as f:
                pickle.dump(data, f)
            f.close()
        return data

    return wrapper


def final_save(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract the true number of topology triangles from the arguments to name the pickle file
        func_params = func.__code__.co_varnames
        if "true_prob_T" in func_params:
            index = func_params.index("true_prob_T")
            if index < len(args):
                prob_T = args[index]
            else:
                prob_T = kwargs.get("true_prob_T", None)

            if prob_T is None:
                raise ValueError(
                    "Parameter 'true_prob_T' not found in function arguments."
                )
        elif "init_params" in func_params:
            index = func_params.index("init_params")
            if index < len(args):
                prob_T = args[index]
            else:
                prob_T = kwargs["init_params"].get("true_prob_T", None)

        else:
            raise ValueError(
                "Parameter 'true_prob_T' not defined in function signature."
            )

        if "simulation_params" in func_params:
            d = kwargs["simulation_params"].get("true_dictionary_type", "separated")
            mode = kwargs["simulation_params"].get("sparsity_mode", "max")

        res, models = func(*args, **kwargs)
        dir = "max_sparsity" if mode == "max" else "random_sparsity"
        file_path = f"results\\final\\{dir}\\res_{d}_T{int(prob_T*100)}.pkl"

        with open(file_path, "wb") as file:
            pickle.dump(models, file)
            pickle.dump(res, file)

        return res, models, file_path

    return wrapper


def save_plot(func):
    """
    Decorator to save the plot returned by the wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        func_name = func.__name__
        plot = func(*args, **kwargs)
        path = os.getcwd()
        dir_name = "plots\\final"

        if func_name == "plot_error_curves":
            d = kwargs.get("dictionary_type", "separated")
            p = kwargs.get("prob_T", 1)
            te = kwargs.get("test_error", True)
            mode = kwargs.get("sparsity_mode", "max")
            dir_name += "\\max_sparsity" if mode == "max" else "\\random_sparsity"
            file_name = (
                f"test_error_{d}_T{int(p*100)}.png"
                if te
                else f"train_error_{d}_T{int(p*100)}.png"
            )

        else:
            return plot

        # file_name = os.path.join(dir_name, file_name)
        file_path = os.path.join(path, dir_name, file_name)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")

        return plot

    return wrapper