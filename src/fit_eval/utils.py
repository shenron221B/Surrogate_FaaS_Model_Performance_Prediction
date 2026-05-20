# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------

import numpy as np
import pickle
import re
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


def get_performances(Xval, Uval, Rval, predictor, train_time, model, base, sim_train_time = None, input_samples = 1000):
    _ = predictor(Xval)  # warm-up

    t2 = time.time()
    Rpred, Upred = predictor(Xval)
    t3=time.time()

    # Confronto
    Umse, Umape, Ur2 = get_metrics(Uval, Upred[:, :len(model.serv_times)])
    Rmse, Rmape, Rr2 = get_metrics(Rval, Rpred[:, :len(model.serv_times)])

    test_time = (t3-t2)/Xval.shape[0]

    return ({
        "Flows": len(model.serv_times),
        "Queue": model.queue_capacity,
        "Id": base,
        "Model": "NNMultiOut",
        "InputSamples": input_samples,
        "TrainingTime": train_time,
        "SimulationTrainingTime": sim_train_time,
        "TestTime": test_time,
        "Umape": Umape,
        "Umse": Umse,
        "Ur2": Ur2,
        "Rmape": Rmape,
        "Rmse": Rmse,
        "Rr2": Rr2
    })

def make_fit_func(strategy, max_functions=None):
    if strategy == "padding":
        from fit_eval.tf_fit import fit_multiout_nn_incremental_padding
        def fit_func(model, X, Y, Y2, hidden_units=None, nn=None, device=None, parallelism = None):
            return fit_multiout_nn_incremental_padding(
                model, X, Y, Y2,
                max_functions=max_functions,
                hidden_units=hidden_units,
                device=device,
                parallelism=parallelism,
                nn=nn
            )
    elif strategy == "copy_layers":
        from fit_eval.tf_fit import fit_multiout_nn_incremental_copy_layers
        def fit_func(model, X, Y, Y2, hidden_units=None, nn=None, device=None, parallelism = None):
            return fit_multiout_nn_incremental_copy_layers(
                model, X, Y, Y2,
                hidden_units=hidden_units,
                device=device,
                parallelism=parallelism,
                nn=nn
            )
    elif strategy == "expand":
        from fit_eval.tf_fit import fit_multiout_nn_incremental_expand
        def fit_func(model, X, Y, Y2, hidden_units=None, nn=None, device=None, parallelism = None):
            return fit_multiout_nn_incremental_expand(
                model, X, Y, Y2,
                hidden_units=hidden_units,
                device=device,
                parallelism=parallelism,
                nn=nn
            )
    else:
        raise ValueError(f"Strategia sconosciuta: {strategy}")
    return fit_func


# Calcola il coefficiente di arrivo per ciascun flusso
def XtoRho(model, X):
    N = len(model.serv_times)
    util_mat = np.array(model.mem_demands) * np.array(model.serv_times) / model.memory  # shape (N,)

    # Moltiplica solo le prime N colonne
    X_rho = np.zeros_like(X)
    X_rho[:, :N] = X[:, :N] * util_mat  # broadcasting automatico

    return X_rho

def get_metrics (Y, Ypred):
    mape = mean_absolute_percentage_error(Y, Ypred)*100
    mse = mean_squared_error(Y, Ypred)
    r2 = r2_score(Y, Ypred)
    return mse, mape, r2


def prepare_dataset(data_dir, base, input_dim=1.0, test_size = 0.2):     #input_dim specifica la percentuale di righe da prendere
    loaded = np.load(f"{data_dir}/{base}.npz")
    X = loaded["X"]
    R = loaded["RT"]
    U = loaded["U"]

    if isinstance(test_size,float) or int(0.8 * input_dim * 1000) == 800:
        # Prendi solo la percentuale desiderata
        n_rows = int(len(X) * input_dim)
        X = X[:n_rows]
        R = R[:n_rows]
        U = U[:n_rows]


        Xtrain, Xval, Rtrain, Rval, Utrain, Uval = train_test_split(
            X, R, U, test_size=test_size)
    
    elif isinstance(test_size,int):
        X_rest, Xval, R_rest, Rval, U_rest, Uval = train_test_split(
            X, R, U, test_size=test_size)
        

        Xtrain, _, Rtrain, _, Utrain, _ = train_test_split(
            X_rest, R_rest, U_rest,
            train_size=int(0.8 * input_dim * 1000)
        )

    return Xtrain, Xval, Rtrain, Rval, Utrain, Uval


def prepare_model(data_dir,base):
    with open(f"{data_dir}/{base}.model","rb") as of:
        return pickle.load(of)


def _estract_m(filename):
    match = re.search(r"_m(\d+)", filename)
    return int(match.group(1)) if match else None

def order_files_by_seed(data_dir, f_num, qlen, order_by_seed):
    files = []

    for f in os.listdir(data_dir):
        if not f.startswith(f"f{f_num}_q{qlen}") or not f.endswith(".npz"):
            continue

        m_val = _estract_m(f)
        if m_val in order_by_seed:
            files.append((order_by_seed.index(m_val), f))  # usa l'indice per ordinare

    # Ordina i file secondo la lista custom
    files.sort()
    # Restituisce solo i nomi dei file, ordinati
    return [f for _, f in files]

# Padding X per arrivare a max_functions colonne
def pad_X(X, max_functions):
    if X.shape[1] < max_functions:
        pad_width = max_functions - X.shape[1]
        X_pad = np.pad(X, ((0, 0), (0, pad_width)), mode='constant')
    elif X.shape[1] > max_functions:
        raise ValueError(f"X ha {X.shape[1]} colonne, ma max_functions = {max_functions}")
    else:
        X_pad = X
    return X_pad

# Padding degli output per arrivare a output_dim colonne
def pad_Y(Y,Y2,output_dim):
    Y_full = np.concatenate((Y, Y2), axis=1)
    if Y_full.shape[1] < output_dim:
        pad_width = output_dim - Y_full.shape[1]
        Y_full = np.pad(Y_full, ((0, 0), (0, pad_width)), mode='constant')
    elif Y_full.shape[1] > output_dim:
        raise ValueError(f"Output ha {Y_full.shape[1]} colonne, ma 2N = {output_dim}")
    return Y_full