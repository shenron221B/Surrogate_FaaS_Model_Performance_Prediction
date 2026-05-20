import sys
import os
import yaml
import numpy as np
import copy
import pickle
import time

sys.path.append(os.path.join(os.getcwd(), ''))
from models.model import model_from_conf
from data.utils import go_simulate
from utils.time_logger import log_execution_time

SEED = 42

# costanti infrastrutturali
CPU_CORES = 8
CPU_DEMAND = 1.0


def print_usage():
    print("Uso corretto: python3 run_simulation.py <dir_configurazione> <max_req> <num_scenari>")
    print("Esempio: python3 run_simulation.py /root/tesi_project/serverledge/results/.../poisson_600 5.0 10000")
    sys.exit(1)

def calculate_rho(X_row, config_dict):
    """calcola il rho totale di una riga di carico"""
    serv_times = config_dict.get('serv_time_duration', [])
    mem_demands = config_dict.get('mem_demands', [])
    system_memory = config_dict.get('system_memory', 2048)
    
    total_rho = 0.0
    for f in range(min(len(X_row), len(serv_times))):
        rho_f = (X_row[f] * mem_demands[f] * serv_times[f]) / system_memory
        total_rho += rho_f
    return total_rho

def main():
    start_time = time.time()
    if len(sys.argv) < 4:
        print_usage()

    config_dir = sys.argv[1]
    try:
        max_req = float(sys.argv[2])
        num_samples = int(sys.argv[3])
    except ValueError:
        print("Errore: max_req deve essere float (es. 5.0) e num_scenari intero (es. 10000).")
        sys.exit(1)

    conf_file = os.path.join(config_dir, "simulator-conf.yml")
    if not os.path.exists(conf_file):
        print(f"[ERRORE] File di configurazione non trovato in {conf_file}")
        sys.exit(1)

    with open(conf_file, 'r') as f:
        config_data = yaml.safe_load(f)

    # costruzione modello base da cui il simulatore farà le copie
    base_model = model_from_conf(config_data)

    num_funcs = len(base_model.mem_demands)

    yaml_serv_times = getattr(base_model, 'serv_times', config_data.get('serv_time_duration', [0.2] * num_funcs))
    base_model.deadlines = [st * 2.5 for st in yaml_serv_times]

    print(f"\n{'=' * 60}")
    print(f" AVVIO SIMULAZIONE SINTETICA UNIFICATA ({num_funcs} FUNZIONI)")
    print(f" Cartella Input/Output : {config_dir}")
    print(f" Scenari Random base   : {num_samples}")
    print(f" Carico Max per f()    : {max_req} req/s")
    print(f"{'=' * 60}\n")

    # generazione carichi sintetici
    rng = np.random.default_rng(SEED)
    X_random = rng.uniform(0.05, max_req, (num_samples, num_funcs))

    # iniezione dataset reale
    real_dataset_path = os.path.join(config_dir, "dataset.npz")
    if os.path.exists(real_dataset_path):
        print("[INFO] Trovato dataset reale! Estrazione carichi per iniezione...")
        real_data = np.load(real_dataset_path)
        X_real = real_data['X']

        if len(X_real.shape) == 1:
            X_real = X_real.reshape(-1, 1)
        X_real = X_real[:, :num_funcs]

        X_synthetic = np.vstack((X_real, X_random))
        print(f"[INFO] {len(X_real)} carichi reali aggiunti con successo alla simulazione.")
    else:
        print(f"[WARN] File {real_dataset_path} non trovato. Generazione solo casuale.")
        X_synthetic = X_random

    total_samples = len(X_synthetic)
    print(f"\n[INFO] Totale scenari da simulare: {total_samples}")

    init_model_path = os.path.join(config_dir, "init_rho_model.pkl") # Usa il nuovo nome!
    predicted_inits = None
    if os.path.exists(init_model_path):
        print("[INFO] Trovato init_rho_model.pkl! Predizione InitTime da Rho in corso...")
        with open(init_model_path, "rb") as f:
            init_predictor = pickle.load(f)
            
        # calcolo del rho per tutti gli scenari 
        rho_synthetic = np.array([calculate_rho(x, config_data) for x in X_synthetic]).reshape(-1, 1)
        
        # predizione degli init usando Rho
        predicted_inits = init_predictor.predict(rho_synthetic)
    else:
        print("[WARN] init_time_model.pkl non trovato. Uso i valori statici dal YAML.")

    # preparazione modelli da simulare
    models_to_simulate = []
    for i in range(total_samples):
        m = copy.deepcopy(base_model)
        m.arv_rates = X_synthetic[i, :].tolist()

        if predicted_inits is not None:
            # Sostituzione degli init time statici con quelli predetti
            m.init_times = predicted_inits[i, :].tolist()
            if i < 3: print(f"[DEBUG] Carico: {m.arv_rates} -> Init Dinamico: {m.init_times}")
        models_to_simulate.append(m)

    # esecuzione simulatore Go
    print(f"Avvio simulazione Go in corso (Attendere)...")
    results = go_simulate(models_to_simulate, n_arrivals=100000, seed=SEED, parallelism=1)

    if not results:
        print("[ERRORE] Nessun risultato ritornato dal simulatore Go.")
        return

    # costruzione feature
    RT = np.zeros((total_samples, num_funcs))
    U = np.zeros((total_samples, num_funcs))
    Success = np.zeros((total_samples, num_funcs))
    Cold = np.zeros((total_samples, num_funcs))
    Warm = np.zeros((total_samples, num_funcs))
    Queue = np.zeros((total_samples, num_funcs))

    # feature costanti vettoriali
    Init = np.zeros((total_samples, num_funcs))
    NetOv = np.zeros((total_samples, num_funcs))
    PoolMem = np.zeros((total_samples, num_funcs))
    FuncMem = np.zeros((total_samples, num_funcs))
    Cpus = np.zeros((total_samples, num_funcs))
    FuncCpu = np.zeros((total_samples, num_funcs))
    QueueLen = np.zeros((total_samples, num_funcs))

    # estrazione parametri infrastrutturali dal modello
    yaml_net_ov = getattr(base_model, 'net_overhead', config_data.get('net_overhead', 0.0))
    yaml_init = getattr(base_model, 'init_times', config_data.get('init_times', [0.0] * num_funcs))
    yaml_qlen = getattr(base_model, 'queue_capacity', config_data.get('qlen', 5))
    yaml_sysmem = getattr(base_model, 'memory', config_data.get('system_memory', 2048))
    yaml_funcmem = getattr(base_model, 'mem_demands', config_data.get('mem_demands', [256] * num_funcs))

    for i, res in enumerate(results):
        RT[i, :] = res["AvgRT"]
        U[i, :] = res["Utility"]

        completions = np.array(res["Completions"])
        arrivals = np.array(res["Arrivals"])
        cold_starts = np.array(res["ColdStarts"])

        # Success Rate
        with np.errstate(divide='ignore', invalid='ignore'):
            succ_rate = completions / arrivals
            succ_rate[arrivals == 0] = 1.0  # nessuna richiesta = 100% success base
        Success[i, :] = succ_rate

        # cold e warm rate
        with np.errstate(divide='ignore', invalid='ignore'):
            cold_rate = cold_starts / completions
            cold_rate[completions == 0] = 0.0
        Cold[i, :] = cold_rate
        Warm[i, :] = 1.0 - cold_rate
        Warm[i, completions == 0] = 0.0

        # derivazione inversa del tempo in coda
        for f in range(num_funcs):
            # Coda = RT_Tot - Esecuzione - Rete - (Probabilità_Cold * Overhead_Accensione)
            # avg_queue = RT[i, f] - yaml_serv_times[f] - yaml_net_ov - (cold_rate[f] * yaml_init[f])
            # Queue[i, f] = max(0.0, avg_queue)  # max per evitare valori negativi dovuti ad approssimazioni

            # riempimento feature costanti
            # Init[i, f] = yaml_init[f]
            
	        # estrazione degli init_time non costanti
            actual_init_f = models_to_simulate[i].init_times[f]
            avg_queue = RT[i, f] - yaml_serv_times[f] - yaml_net_ov - (cold_rate[f] * actual_init_f)
            Queue[i, f] = max(0.0, avg_queue)
            Init[i, f] = actual_init_f

            NetOv[i, f] = yaml_net_ov
            PoolMem[i, f] = yaml_sysmem
            FuncMem[i, f] = yaml_funcmem[f]
            Cpus[i, f] = CPU_CORES
            FuncCpu[i, f] = CPU_DEMAND
            QueueLen[i, f] = yaml_qlen

    # salvataggio
    max_req_str = str(max_req).replace(".", "_")
    npz_filename = f"dataset_simulated_{max_req_str}_{num_samples}.npz"
    pkl_filename = f"base_model_{max_req_str}_{num_samples}.pkl"

    output_npz = os.path.join(config_dir, npz_filename)
    output_pkl = os.path.join(config_dir, pkl_filename)

    np.savez(output_npz, X=X_synthetic, RT=RT, U=U,
             Warm=Warm, Cold=Cold, Success=Success, Queue=Queue,
             Init=Init, NetOv=NetOv, PoolMem=PoolMem, FuncMem=FuncMem,
             Cpus=Cpus, FuncCpu=FuncCpu, QueueLen=QueueLen)

    with open(output_pkl, "wb") as f:
        pickle.dump(base_model, f)
    
    end_time = time.time()
    total_sim_time = end_time - start_time
    log_execution_time(config_dir, "Simulatore", total_sim_time)

    print(f"\n[OK] Simulazione completata con successo!")
    print(f"Dataset salvato in: {output_npz} (Con {len(np.load(output_npz).files)} feature ML!)")
    print(f"RT Medio globale: {np.mean(RT):.4f} s")
    print(f"Utility Media globale: {np.mean(U):.4f}")
    print(f"Success Rate globale: {np.mean(Success) * 100:.2f}%")

if __name__ == "__main__":
    main()
