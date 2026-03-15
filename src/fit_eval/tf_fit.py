import numpy as np
import tensorflow as tf

from sklearn.preprocessing import PolynomialFeatures
from models.tf_model_defs import expand_nn, build_nn, expand_input_layer,expand_output_layer
from fit_eval.utils import XtoRho, pad_X, pad_Y

WARNING = False

def _make_predictor_with_padding(model, poly, nn_model, max_functions):
    def predictor(x):
        if x.shape[1] < max_functions:
            pad_width = max_functions - x.shape[1]
            x_pad = np.pad(x, ((0, 0), (0, pad_width)), mode='constant')
        elif x.shape[1] > max_functions:
            raise ValueError(f"x ha {x.shape[1]} colonne, ma max_functions = {max_functions}")
        else:
            x_pad = x

        x_rho = XtoRho(model, x_pad)
        x_poly = poly.transform(x_rho)
        return np.hsplit(nn_model.predict(x_poly, verbose=0), 2)

    return predictor


def _make_predictor(model, poly, nn_model):
    def predictor(x):

        x_rho = XtoRho(model, x)
        x_poly = poly.transform(x_rho)
        return np.hsplit(nn_model.predict(x_poly, verbose=0), 2)

    return predictor

def _parallelism(device, parallelism = None):
    global WARNING
    if device == "cpu":
        if parallelism is not None and WARNING == False: 
            tf.config.threading.set_intra_op_parallelism_threads(parallelism)
            tf.config.threading.set_inter_op_parallelism_threads(2)
            print(f"[WARNING] Parallelismo impostato a {parallelism}")
            WARNING = True
        elif WARNING == False:
            import multiprocessing
            parallelism = multiprocessing.cpu_count()
            print(f"[WARNING] Parallelismo non impostato, verranno utilizzati tutti i core della CPU: {parallelism}")
            WARNING = True
    elif device == "gpu":
        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            device_to_use = '/GPU:0'
            tf.device(device_to_use)
            if WARNING == False:
                print("[WARNING] Il training utilizza la GPU")
                WARNING = True
        else:
            raise ValueError("[ERROR] Nessuna GPU rilevata")
    else: 
        raise ValueError("[ERROR] Device specificato non valido: cpu o gpu ")

# Addestramento rete neurale con un sola uscita per funzione
def fit_nn(model, X, Y, hidden_units = None, l2reg=0.00005, nn=None, device = "cpu", parallelism = None):
    _parallelism(device, parallelism=parallelism)

    N = len(model.serv_times)

    # Trasformazione degli input includendo interazioni non lineari fino al grado 3
    X2 = XtoRho(model, X)
    poly = PolynomialFeatures(3)
    X2 = poly.fit_transform(X2)

    nn = build_nn(N, hidden_units, l2reg)
    nn.fit(X2, Y, validation_split=0.2, batch_size=16, verbose = 0,
           epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])
    return lambda x: nn.predict(poly.transform(XtoRho(model, x)), verbose=0), nn


# Questa rete è in grado di predire più feature per una singola funzione
def fit_multiout_nn(model, X, Y, Y2, hidden_units = None, l2reg=0.0001, nn=None, device = "cpu", parallelism = None):
    _parallelism(device, parallelism=parallelism)

    N = len(model.serv_times)

    # Trasformazione degli input includendo interazioni non lineari fino al grado 3
    X2 = XtoRho(model, X)
    poly = PolynomialFeatures(3)
    X2 = poly.fit_transform(X2)

    Y_full = np.concatenate((Y, Y2), axis=1)

    # Layer di output di dimensione 2N, in questo modo produce due output per funzione
    if not nn and hidden_units is not None:
        nn = build_nn(2*N, hidden_units, l2reg)

    nn.fit(X2, Y_full, validation_split=0.2, batch_size=16, verbose = 0,
           epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])


    predictor = _make_predictor(model, poly, nn)
    return nn, predictor


def fit_multiout_nn_incremental_padding(model, X, Y, Y2, max_functions, hidden_units = None, nn = None, l2reg=0.0001, device = "cpu", parallelism = None):
    _parallelism(device, parallelism=parallelism)

    output_dim = 2 * max_functions                           # output: 2 per funzione 

    # Padding X 
    X_pad = pad_X(X,max_functions)

    # Trasformazione non lineare
    X_rho = XtoRho(model, X_pad)                  
    poly = PolynomialFeatures(3)
    X2 = poly.fit_transform(X_rho)               

    # Padding degli output
    Y_full = pad_Y(Y,Y2,output_dim)


    if not nn and hidden_units is not None:
        # Costruzione della rete con il padding
        nn = build_nn(output_dim, hidden_units, l2reg)

    # Addestramento 
    nn.fit(X2, Y_full,
           validation_split=0.2,
           batch_size=16,
           epochs=100,
           verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])

    predictor = _make_predictor_with_padding(model, poly, nn, max_functions)
    return nn, predictor

def fit_multiout_nn_incremental_copy_layers(model, X, Y, Y2, hidden_units=None, nn=None, l2reg=0.0001, device = "cpu", parallelism = None):
    _parallelism(device, parallelism=parallelism)

    # Numero funzioni attuali
    n_functions = len(model.serv_times)
    output_dim = 2 * n_functions

    # Trasformazione feature
    X_rho = XtoRho(model, X)                  
    poly = PolynomialFeatures(3)
    X2 = poly.fit_transform(X_rho)
    input_dim = X2.shape[1]

    # Prepara output
    Y_full = np.hstack([Y, Y2])

    # Crea o espandi rete
    nn = expand_nn(nn, input_dim, output_dim, hidden_units, l2reg)

    # Addestramento
    nn.fit(X2, Y_full,
           validation_split=0.2,
           batch_size=16,
           epochs=100,
           verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])

    predictor = _make_predictor(model, poly, nn)
    return nn, predictor


def fit_multiout_nn_incremental_expand(model, X, Y, Y2, hidden_units=None, nn=None, l2reg=0.0001, device = "cpu", parallelism = None):
    _parallelism(device, parallelism=parallelism)

    # Numero funzioni attuali
    n_functions = len(model.serv_times)
    output_dim = 2 * n_functions

    # Trasformazione feature
    X_rho = XtoRho(model, X)                  
    poly = PolynomialFeatures(3)
    X2 = poly.fit_transform(X_rho)
    input_dim = X2.shape[1]

    # Prepara output
    Y_full = np.hstack([Y, Y2])

    if not nn and hidden_units is not None:
        nn = build_nn(output_dim, hidden_units, l2reg)
    else:
        # Crea o espandi rete
        new_input_layer = expand_input_layer(nn.layers[0], input_dim)
        new_output_layer = expand_output_layer(nn.layers[-1], output_dim)
        nn = tf.keras.Sequential([new_input_layer] + nn.layers[1:-1]+[new_output_layer])
        nn.compile(optimizer='adam', loss='mse')

    # Addestramento
    nn.fit(X2, Y_full,
           validation_split=0.2,
           batch_size=16,
           epochs=100,
           verbose=0,
           callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss")])

    predictor = _make_predictor(model, poly, nn)
    return nn, predictor