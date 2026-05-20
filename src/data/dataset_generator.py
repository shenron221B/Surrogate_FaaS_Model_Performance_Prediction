# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------


import numpy as np
import pickle
import os
import copy

from data.utils import create_random_X, go_simulate, add_random_X
from models.model import random_model, add_random_functions_to_model, model_from_conf

MAX_ARRIVALS=5*10e5

# Simula più scenari diversi definite dalle righe di X 
def _simulateX (model, X, max_arrivals=MAX_ARRIVALS, parallelism = None):
    n = len(model.serv_times)

    # Models to feed as input to the simulator
    models = []
    for i in range(X.shape[0]):
        models.append(copy.copy(model))
        models[i].arv_rates = X[i,:]
    
    # Lancia il simulatore in Go e prende i risultati
    results = go_simulate(models, int(max_arrivals), parallelism=parallelism)

    output = np.zeros((X.shape[0], n))          # Tempo medio di risposta
    outputU = np.zeros((X.shape[0], n))         # Utilizzo
    outputCold = np.zeros((X.shape[0], n))      # Frequenza di cold starts
    tput = np.zeros((X.shape[0], n))            # Throughput
    
    # Carichiamo e ritorniamo il risultato
    for i in range(X.shape[0]):
        output[i,:] = results[i]["AvgRT"]
        outputU[i,:] = results[i]["Utility"]
        cstarts = np.array(results[i]["ColdStarts"])
        compl = np.array(results[i]["Completions"])
        outputCold[i,:] = cstarts/compl
        tput[i,:] = compl/float(results[i]["Time"])

    return output, outputU, outputCold,tput


def generate_dataset_with_flows_and_seed(out_dir, config, flows,seed, samples=1000):
    import time
    deadline_coeff = config["deadline_coeff"]
    qlen = config["qlen"]
    serv_time_exp = config["serv_time_exp"]
    queue_policy = config["queue_policy"]
    poisson_arrivals = config["poisson_arrivals"]
    init_times = config["init_times"]
    parallelism = config["parallelism"]

    if not os.path.exists(f"{out_dir}/"):
        os.makedirs(out_dir)

    print(f"[INFO] Simulation with [f={flows}, q={qlen}, s={seed}, dc={deadline_coeff}, policy = {queue_policy}, serv_time_exp = {serv_time_exp}, poisson_arrivals = {poisson_arrivals}, init_times = {init_times}]")
    t0 = time.time()
    _base_simulation(flows, qlen, seed, deadline_coeff, out_dir, samples = samples, serv_time_exp=serv_time_exp,queue_policy=queue_policy, poisson_arrivals = poisson_arrivals, parallelism = parallelism, init_times=init_times)
    t1 = time.time()
    simulation_time = t1-t0

    return simulation_time


def generate_dataset_incremental_with_flows_and_seed(data_dir, config, flows,seed, samples=1000):
    import time
    deadline_coeff = config["deadline_coeff"]
    qlen = config["qlen"]
    serv_time_exp = config["serv_time_exp"]
    queue_policy = config["queue_policy"]
    poisson_arrivals = config["poisson_arrivals"]
    init_times = config["init_times"]
    parallelism = config["parallelism"]

    if not os.path.exists(f"{data_dir}/"):
        os.makedirs(data_dir)

    print(f"[INFO] Incrementing function number in dataset {data_dir}/f{flows}_q{qlen}_m{seed}")
    t0 = time.time()
    with open(f"{data_dir}/f{flows}_q{qlen}_m{seed}.model","rb") as of:
        model = pickle.load(of)
    
    loaded = np.load(f"{data_dir}/f{flows}_q{qlen}_m{seed}.npz")
    X=loaded["X"]

    _add_function_and_simulate(model, X, qlen, seed, deadline_coeff, data_dir, samples = samples, poisson_arrivals=poisson_arrivals, init_times=init_times, parallelism=parallelism)
    t1 = time.time()
    print(f"[INFO] Incrementation completed, created dataset: [f={flows+1}, q={qlen}, s={seed}, dc={deadline_coeff}, policy = {queue_policy}, serv_time_exp = {serv_time_exp}, poisson_arrivals = {poisson_arrivals}, init_times = {init_times}]")
    simulation_time = t1-t0

    return simulation_time

def generate_dataset_from_config(data_dir,config, X):
    seeds = config['seeds']
    parallelism = config['parallelism']


    print(f"[INFO] Simulation with config")
    for seed in seeds:
        print(f"[INFO] Simulation with seed: {seed}")
        _simulate_with_config(seed,config,parallelism, data_dir, X)
        print(f"[INFO] Simulation ended, file saved in {data_dir}")
    

def _base_simulation(flows,qlen,seed, deadline_coeff, outdir, samples = 1000, serv_time_exp = True, queue_policy = "fifo", poisson_arrivals=False, init_times = False, parallelism = None):

    rng = np.random.default_rng(seed)

    # Generiamo randomicamente la caratteristiche del modello
    model = random_model(rng, n=flows, serv_time_exp=serv_time_exp, queue_policy=queue_policy, deadline_coeff=deadline_coeff, queue_cap=qlen, poisson_arrivals=poisson_arrivals, no_init_times= not init_times)   

    # Generiamo i tassi di arrivo al sistema
    X = create_random_X(rng, model, samples, max_rho=1.1)

    # Lanciamo la simulazione in Go
    Y, U, C, T = _simulateX(model, X, MAX_ARRIVALS, parallelism = parallelism)
    np.savez(f"{outdir}/f{flows}_q{qlen}_m{seed}.npz", X=X, RT=Y, U=U, C=C, T=T)
    with open(f"{outdir}/f{flows}_q{qlen}_m{seed}.model","wb") as of:
        pickle.dump(model, of)

def _simulate_with_config(seed,config,parallelism, outdir, X):
    model = model_from_conf(config)
    # Lanciamo la simulazione in Go
    Y, U, C, T = _simulateX(model, X, MAX_ARRIVALS, parallelism = parallelism)
    np.savez(f"{outdir}/m{seed}.npz", X=X, RT=Y, U=U, C=C, T=T)
    with open(f"{outdir}/m{seed}.model","wb") as of:
        pickle.dump(model, of)


def _add_function_and_simulate(model, X,qlen,seed, deadline_coeff, outdir, samples = 1000, serv_time_exp = True, poisson_arrivals=True, init_times = False, parallelism = None):

    rng = np.random.default_rng(seed)

    # Aggiungiamo funzioni con parametri casuali al modello
    model = add_random_functions_to_model(model, rng, deadline_coeff=deadline_coeff, serv_time_exp = serv_time_exp, poisson_arrivals=poisson_arrivals, no_init_times= not init_times) 

    # Generiamo i tassi di arrivo al sistema
    X = add_random_X(rng, model, X, samples = samples, max_rho=1.1)

    # Lanciamo la simulazione in Go
    Y, U, C, T = _simulateX(model, X, MAX_ARRIVALS, parallelism=parallelism)

    np.savez(f"{outdir}/f{len(model.serv_times)}_q{qlen}_m{seed}.npz", X=X, RT=Y, U=U, C=C, T=T)
    with open(f"{outdir}/f{len(model.serv_times)}_q{qlen}_m{seed}.model","wb") as of:
        pickle.dump(model, of)

