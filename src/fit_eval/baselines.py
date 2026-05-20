import numpy as np
import time
from fit_eval.utils import get_metrics,prepare_dataset, prepare_model, order_files_by_seed

def evaluate_mmck(data_dir, config):
    import models.mmck as mmck
    # Configurazione
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]

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

            # Modello analitico che non ha bisogno di addestramento
            mmckModel = mmck.MMckModel(model, partitioning_mode_alternative=True)

            # warm up
            _,_ = mmckModel.predict(Xval)

            t0=time.time()
            Rpred, Upred = mmckModel.predict(Xval)
            t1=time.time()

            Rpred = np.nan_to_num(Rpred)
            Upred = np.nan_to_num(Upred)
            # Confronto tra valori predetti e valori reali
            Umse,Umape,Ur2 = get_metrics(Uval,Upred)
            Rmse,Rmape,Rr2 = get_metrics(Rval,Rpred)
            test_time = (t1-t0)/Xval.shape[0]
            
            results.append({
                    "Flows": len(model.serv_times),
                    "Queue": model.queue_capacity,
                    "Id": base,
                    "Model": "MMck-alt",
                    "InputSamples": 1000,
                    "TrainingTime": 0,
                    "TestTime": test_time,
                    "Umape": Umape,
                    "Umse": Umse,
                    "Ur2": Ur2,
                    "Rmape": Rmape,
                    "Rmse": Rmse,
                    "Rr2": Rr2
                })
    return results
    
def evaluate_kaufman(data_dir, config):
    from models.kaufman import ExactKaufman
    # Configurazione
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]

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

            kaufman = ExactKaufman(model)

            # warm up
            _, _ = kaufman.predict(Xval)

            t0=time.time()
            kaufmanR, kaufmanU = kaufman.predict(Xval)
            t1=time.time()

            # Confronto tra valori predetti e valori reali
            Umse,Umape,Ur2 = get_metrics(Uval,kaufmanU)
            Rmse,Rmape,Rr2 = get_metrics(Rval,kaufmanR)
            test_time = (t1-t0)/Xval.shape[0]
        
            results.append({
                "Flows": len(model.serv_times),
                "Queue": model.queue_capacity,
                "Id": base,
                "Model": "Kaufman",
                "InputSamples": 1000,
                "TrainingTime": 0,
                "TestTime": test_time,
                "Umape": Umape,
                "Umse": Umse,
                "Ur2": Ur2,
                "Rmape": Rmape,
                "Rmse": Rmse,
                "Rr2": Rr2
            })

    return results

def evaluate_poly(data_dir, config, deg=3):
    from models.poly import fit_poly

    # Configurazione
    start_functions = config["start_functions"]
    max_functions = config["max_functions"]
    qlen = config["qlen"]
    seeds = config["seeds"]

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

            # Addestramento
            t0 = time.time()
            Rpoly = fit_poly(model, Xtrain, Rtrain, deg)
            Upoly = fit_poly(model, Xtrain, Utrain, deg)
            t1 = time.time()
            # Confrontiamo i valori predetti con i valori reali
            Umse,Umape,Ur2 = get_metrics(Upoly(Xval), Uval)
            Rmse,Rmape,Rr2 = get_metrics(Rpoly(Xval), Rval)
            t2 = time.time()
            train_time = t1-t0
            test_time = (t2-t1)/Xval.shape[0]


            results.append({
                "Flows": len(model.serv_times),
                "Queue": model.queue_capacity,
                "Id": base,
                "Model": "Kaufman",
                "InputSamples": 1000,
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