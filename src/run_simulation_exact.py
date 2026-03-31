import sys
import os
import yaml
import numpy as np
import copy
import pickle

sys.path.append(os.path.join(os.getcwd(), 'src'))
from models.model import model_from_conf
from data.utils import go_simulate

SEED = 42
CPU_CORES = 8
CPU_DEMAND = 1.0

def print_usage():
    print("Uso: python3 run_simulation_exact.py <dir_configurazione>")
    print("Esempio: python3 run_simulation_exact.py /root/.../poisson_600")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print_usage()

    config_dir = os.path.abspath(sys.argv[1])
    conf_file = os.path.join(config_dir, "simulator-conf.yml")
    real_dataset_path = os.path.join(config_dir, "dataset.npz")

    if not os.path.exists(conf_file):
        print(f"[ERRORE] File YAML non trovato: {conf_file}")
        sys.exit(1)
    if not os.path.exists(real_dataset_path):
        print(f"[ERRORE] Dataset reale non trovato: {real_dataset_path}")
        sys.exit(1)

    with open(conf_file, 'r') as f:
        config_data = yaml.safe_load(f)

    base_model = model_from_conf(config_data)
    num_funcs = len(base_model.mem_demands)
    yaml_serv_times = getattr(base_model, 'serv_times', config_data.get('serv_time_duration', [0.2] * num_funcs))
    base_model.deadlines = [st * 2.5 for st in yaml_serv_times]

    print(f"\n{'=' * 60}")
    print(f" AVVIO SIMULAZIONE EXACT MATCH ({num_funcs} FUNZIONI)")
    print(f" Cartella Input/Output : {config_dir}")
    print(f"{'=' * 60}\n")

    # Caricamento solo dei carichi reali
    real_data = np.load(real_dataset_path)
    X_real = real_data['X']
    if len(X_real.shape) == 1:
        X_real = X_real.reshape(-1, 1)
    X_real = X_real[:, :num_funcs]

    total_samples = len(X_real)
    print(f"[INFO] Trovate {total_samples} righe di carico nel dataset reale.")
    print("[INFO] Prime 5 righe estratte per verifica:")
    for i in range(min(5, total_samples)):
        print(f"  Row {i+1}: {X_real[i]}")

    models_to_simulate = []
    for i in range(total_samples):
        m = copy.deepcopy(base_model)
        m.arv_rates = X_real[i, :].tolist()
        models_to_simulate.append(m)

    print(f"\n[INFO] Avvio simulazione Go in corso (Attendere)...")
    results = go_simulate(models_to_simulate, n_arrivals=100000, seed=SEED, parallelism=1)

    if not results:
        print("[ERRORE] Nessun risultato ritornato dal simulatore Go.")
        return

    # Inizializzazione matrici
    RT, U, Success, Cold, Warm, Queue = [np.zeros((total_samples, num_funcs)) for _ in range(6)]
    Init, NetOv, PoolMem, FuncMem, Cpus, FuncCpu, QueueLen = [np.zeros((total_samples, num_funcs)) for _ in range(7)]

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

        with np.errstate(divide='ignore', invalid='ignore'):
            succ_rate = completions / arrivals
            succ_rate[arrivals == 0] = 1.0
            cold_rate = cold_starts / completions
            cold_rate[completions == 0] = 0.0

        Success[i, :] = succ_rate
        Cold[i, :] = cold_rate
        Warm[i, :] = 1.0 - cold_rate
        Warm[i, completions == 0] = 0.0

        for f in range(num_funcs):
            avg_queue = RT[i, f] - yaml_serv_times[f] - yaml_net_ov - (cold_rate[f] * yaml_init[f])
            Queue[i, f] = max(0.0, avg_queue)
            Init[i, f] = yaml_init[f]
            NetOv[i, f] = yaml_net_ov
            PoolMem[i, f] = yaml_sysmem
            FuncMem[i, f] = yaml_funcmem[f]
            Cpus[i, f] = CPU_CORES
            FuncCpu[i, f] = CPU_DEMAND
            QueueLen[i, f] = yaml_qlen

    npz_filename = "dataset_simulated_exact.npz"
    output_npz = os.path.join(config_dir, npz_filename)

    np.savez(output_npz, X=X_real, RT=RT, U=U, Warm=Warm, Cold=Cold,
             Success=Success, Queue=Queue, Init=Init, NetOv=NetOv,
             PoolMem=PoolMem, FuncMem=FuncMem, Cpus=Cpus,
             FuncCpu=FuncCpu, QueueLen=QueueLen)

    print(f"\n[OK] Simulazione Exact Match completata!")
    print(f"Dataset salvato in: {output_npz}")

if __name__ == "__main__":
    main()