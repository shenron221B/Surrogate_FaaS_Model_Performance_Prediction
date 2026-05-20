import time
from fit_eval.utils import get_metrics,prepare_dataset, prepare_model, order_files_by_seed, get_performances
from keras import backend as K
import gc
import tensorflow as tf

def evaluate_nn(data_dir, config):
    from fit_eval.tf_fit import  fit_nn

    # Configurazione
    hidden_units = config["hidden_units"]
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]
    device = config["tf_device"]
    parallelism = config["parallelism"]

    print(f"[INFO] Sto caricando i dati da {data_dir}")
    results = [] 
    for function in range (start_functions, max_functions+1):
        # Ordina i file in base al seed
        ordered_files = order_files_by_seed(data_dir, function,qlen,seeds)
        
        for f in ordered_files:
            base, _ = f.split(".")
            print(f"[INFO] Processing {base} with {function} functions")
            Xtrain, Xval, Rtrain, Rval, Utrain, Uval = prepare_dataset(data_dir,base)
            model = prepare_model(data_dir, base)     

            # Addestriamo due reti singole con un solo livello, una per RT ed una per U
            t0=time.time()
            Upredictor, _ = fit_nn(model, Xtrain, Utrain, hidden_units, device=device, parallelism=parallelism)
            Rpredictor, _ = fit_nn(model, Xtrain, Rtrain, hidden_units, device=device, parallelism=parallelism)
            t1=time.time()

            # warm up
            _ = Rpredictor(Xval)
            _ = Upredictor(Xval)

            t2 = time.time()
            Rpred = Rpredictor(Xval)
            Upred = Upredictor(Xval)
            t3 = time.time()

            # Confronto
            Umse,Umape,Ur2 = get_metrics(Uval,Upred)
            Rmse,Rmape,Rr2 = get_metrics(Rval,Rpred)

            train_time = t1-t0
            test_time = (t3-t2)/Xval.shape[0]
            results.append({
                "Flows": len(model.serv_times),
                "Queue": model.queue_capacity,
                "Id": base,
                "Model": "NN",
                "TrainingTime": train_time,
                "TestTime": test_time,
                "Umape": Umape,
                "Umse": Umse,
                "Ur2": Ur2,
                "Rmape": Rmape,
                "Rmse": Rmse,
                "Rr2": Rr2
            })
    return results


def evaluate_multiout_nn(data_dir, config, save_nns = False, get_simulation_time = None):
    from fit_eval.tf_fit import  fit_multiout_nn
    import pandas as pd

    # Configurazione
    hidden_units = config["hidden_units"]
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]
    device = config["tf_device"]
    parallelism = config["parallelism"]

    results = []
    nns = []  # Lista per le reti neurali 

    for function in range (start_functions, max_functions+1):

        print(f"[INFO] Sto caricando i dati da {data_dir}")
        # Ordina i file in base al seed
        ordered_files = order_files_by_seed(data_dir, function,qlen,seeds)
        
        for i,f in enumerate(ordered_files):
            base, _ = f.split(".")
            print(f"[INFO] Processing {base} with {function} functions")
            Xtrain, Xval, Rtrain, Rval, Utrain, Uval = prepare_dataset(data_dir,base)
            model = prepare_model(data_dir, base)        

            # Addestramento
            t0=time.time()
            nn, predictor = fit_multiout_nn(model, Xtrain, Rtrain, Utrain, hidden_units = hidden_units, device=device, parallelism=parallelism)
            t1=time.time()

            train_time = t1-t0
            sim_train_time = None
            if get_simulation_time is not None:
                if isinstance(get_simulation_time, pd.DataFrame):
                    df_filtered = get_simulation_time.loc[
                        (get_simulation_time["seed"] == seeds[i]) &
                        (get_simulation_time["data_dim"] == 1000) &
                        (get_simulation_time["flows"] == function)
                    ]
                    if not df_filtered.empty:
                        sim_time_val = df_filtered["simulation_time"].values[0]
                        sim_train_time = train_time + sim_time_val
                else:
                    raise ValueError("get_simulation_time deve essere un DataFrame pandas")

            if save_nns:
                nns.append(nn)  # Aggiungi la rete alla lista

            results.append(get_performances(Xval,Uval,Rval,predictor,train_time,model,base, sim_train_time=sim_train_time))

    return results, nns


# Questa funzione addestra una rete (specifcando rete e seed) per diverse dimensioni del dataset
def evaluate_monn_with_seed_different_datasize(data_dir, config, nn, num_functions, new_seed, intervals=5, get_simulation_time=None):
    from fit_eval.tf_fit import fit_multiout_nn
    import pandas as pd

    # Configurazione
    qlen = config["qlen"]
    device = config["tf_device"]
    parallelism = config["parallelism"]

    temp_nn = [nn]
    temp_nns = [temp_nn.copy() for _ in range(intervals)]

    interval_size = 1 / intervals
    results = []
    new_seed = [new_seed]
    for idx, k in enumerate(temp_nns, start=1):
        data_size = round(idx * interval_size, 1)

        print(f"[INFO] Sto caricando i dati da {data_dir}")

        ordered_files = order_files_by_seed(data_dir, num_functions, qlen, new_seed)
        base, _ = ordered_files[0].split(".")
        print(f"[INFO] Processing {base} con {int(data_size * 1000)} input data")

        Xtrain, Xval, Rtrain, Rval, Utrain, Uval = prepare_dataset(data_dir, base, input_dim=data_size, test_size=200)
        model = prepare_model(data_dir, base)

        t0 = time.time()
        _, predictor = fit_multiout_nn(model, Xtrain, Rtrain, Utrain, nn=k[0], device=device, parallelism=parallelism)
        t1 = time.time()

        train_time = t1 - t0

        sim_train_time = None
        if get_simulation_time is not None:
            if isinstance(get_simulation_time, pd.DataFrame):
                df_filtered = get_simulation_time.loc[
                    (get_simulation_time["seed"] == new_seed[0]) &
                    (get_simulation_time["flows"] == num_functions)
                ]
                if not df_filtered.empty:
                    sim_time_val = df_filtered["simulation_time"].values[0]
                    sim_time_val *= data_size
                    sim_train_time = sim_time_val + train_time
            else:
                raise ValueError("get_simulation_time deve essere un DataFrame pandas")


        results.append(get_performances(Xval, Uval, Rval, predictor, train_time, model, base, sim_train_time=sim_train_time, input_samples=int(data_size*1000)))

    for nn in temp_nns:
        try:
            del nn
        except:
            pass
    K.clear_session()
    gc.collect()
    return results



def evaluate_multiout_nn_incremental(strategy, data_dir, config, data_dim = 1.0, get_simulation_time = None, test_size = 0.2):
    import pandas as pd
    from fit_eval.utils import make_fit_func

    # Configurazione
    hidden_units = config["hidden_units"]
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]
    device = config["tf_device"]
    parallelism = config["parallelism"]

    fit_func = make_fit_func(strategy,max_functions)
 
    results = []

    # Reti neurali per ogni seed
    nns = []

    print(f"[INFO] Sto caricando i dati da {data_dir}")
    ordered_files = order_files_by_seed(data_dir, start_functions,qlen,seeds)

    # Primo addestramento
    for i, f in enumerate(ordered_files):
        base, _ = f.split(".")
        print(f"[INFO] Processing {base} with {start_functions} functions")
        Xtrain, Xval, Rtrain, Rval, Utrain, Uval = prepare_dataset(data_dir, base)
        model = prepare_model(data_dir, base)
        
        t0 = time.time()
        nn, predictor = fit_func(model, Xtrain, Rtrain, Utrain, hidden_units=hidden_units, device=device, parallelism=parallelism)
        t1 = time.time()

        nns.append(nn)

        train_time = t1 - t0
        sim_train_time = None
        if get_simulation_time is not None:
            if isinstance(get_simulation_time, pd.DataFrame):
                df_filtered = get_simulation_time.loc[
                    (get_simulation_time["seed"] == seeds[i]) &
                    (get_simulation_time["data_dim"] == 1000) &
                    (get_simulation_time["flows"] == start_functions)
                ]
                if not df_filtered.empty:
                    sim_time_val = df_filtered["simulation_time"].values[0]
                    sim_train_time = sim_time_val + train_time
            else:
                raise ValueError("get_simulation_time deve essere un DataFrame pandas")

        results.append(get_performances(Xval, Uval, Rval, predictor, train_time, model, base, sim_train_time=sim_train_time))

    # Addestramento incrementale da start_functions a max_functions
    for function in range(start_functions + 1, max_functions + 1):

        ordered_files = order_files_by_seed(data_dir, function, qlen, seeds)

        for i, f in enumerate(ordered_files):
            base, _ = f.split(".")
            print(f"[INFO] Processing {base} with {function} functions")

            Xtrain, Xval, Rtrain, Rval, Utrain, Uval = prepare_dataset(data_dir, base, input_dim=data_dim, test_size=test_size)
            model = prepare_model(data_dir, base)
            
            t0 = time.time()
            nn, predictor = fit_func(model, Xtrain, Rtrain, Utrain, hidden_units=hidden_units, nn=nns[i], device=device, parallelism=parallelism)
            t1 = time.time()

            train_time = t1 - t0
            sim_train_time = None
            if get_simulation_time is not None:
                if isinstance(get_simulation_time, pd.DataFrame):
                    df_filtered = get_simulation_time.loc[
                        (get_simulation_time["seed"] == seeds[i]) &
                        (get_simulation_time["flows"] == function)
                    ]
                    if not df_filtered.empty:
                        sim_time_val = df_filtered["simulation_time"].values[0]
                        sim_time_val *= data_dim
                        sim_train_time = sim_time_val + train_time
                else:
                    raise ValueError("get_simulation_time deve essere un DataFrame pandas")

            nns[i] = nn  # Aggiorna la rete nella lista
            results.append(get_performances(Xval, Uval, Rval, predictor, train_time, model, base, sim_train_time=sim_train_time, input_samples=int(1000*data_dim)))

    for nn in nns:
        try:
            del nn
        except:
            pass
    K.clear_session()
    gc.collect()
    tf.keras.backend.clear_session()

    return results