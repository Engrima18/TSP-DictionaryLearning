import pickle
import os
import hashlib
from functools import wraps

def memoize(file_name='cache\\DataTS.pkl'):
    '''
    Decorator for caching function results in a pickle file
    and speed up the calculation.
    '''
    def decorator(func):
        def wrapper(self, *args, **kwargs):

            graph_hash = hashlib.sha256(str((sorted(self.edges()), self.id)).encode()).hexdigest()
            key = f"{func.__name__}_{graph_hash}"
            
            if os.path.exists(file_name):
                with open(file_name, 'rb') as file:
                    try:
                        data = pickle.load(file)
                    except EOFError:
                        data = {}
                if key in data:
                    return data[key]

            result = func(self, *args, **kwargs)
            try:
                if os.path.exists(file_name):
                    with open(file_name, 'rb') as file:
                        data = pickle.load(file)
                else:
                    data = {}
            except EOFError:
                data = {}
            data[key] = result
            with open(file_name, 'wb') as file:
                pickle.dump(data, file)

            return result
        return wrapper
    return decorator


def memoize_or_save(func):
    @wraps(func)
    def wrapper(Lu, Ld, **kwargs):
        # Construct file paths based on function arguments
        dictionary_type = kwargs.get('dictionary_type', 'separated')
        prob_T = kwargs.get('prob_T', 1)

        if prob_T == 1:
            name = f'full_data_{dictionary_type}'
        else:
            name = f'top_data_T{int(prob_T*100)}'
        path = os.getcwd()
        dir_path = f'{path}\\synthetic_data'
        filename = f'{dir_path}\\{name}.pkl'

        try:
            # Try to load data from file
            with open(filename, 'rb') as f: 
                data = pickle.load(f)
            f.close()
        except FileNotFoundError:
            # If file not found, generate data and save
            D_true, Y_train, Y_test, X_train, X_test, epsilon_true, c_true = func(Lu, Ld, **kwargs)
            data = {'D_true':D_true,
                    'Y_train':Y_train,
                    'Y_test':Y_test,
                    'X_train':X_train,
                    'X_test':X_test,
                    'epsilon_true':epsilon_true,
                    'c_true':c_true}

            os.makedirs(dir_path, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
            f.close()
        return data

    return wrapper