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


def print_usage():
    print("Uso corretto: python3 run_simulation_2f.py <dir_configurazione> <max_req> <num_scenari>")
    print(
        "Esempio: python3 run_simulation_2f.py /root/tesi_project/serverledge/results/2f_3GBpm/q_len_5/del/del 30.0 10000")
    sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print_usage()

    config_dir = sys.argv[1]
    try:
        max_req = float(sys.argv[2])
        num_samples = int(sys.argv[3])
    except ValueError:
        print("Errore: max_req deve essere float (es. 30.0) e num_scenari intero (es. 10000).")
        sys.exit(1)

    conf_file = os.path.join(config_dir, "simulator-conf.yml")

    if not os.path.exists(conf_file):
        print(f"Errore: File di configurazione non trovato in {conf_file}")
        sys.exit(1)

    with open(conf_file, 'r') as f:
        config_data = yaml.safe_load(f)

    base_model = model_from_conf(config_data)
    num_funcs = len(base_model.mem_demands)

    print(f"\n=======================================================")
    print(f" AVVIO SIMULAZIONE SINTETICA ({num_funcs} FUNZIONI)")
    print(f" Cartella Input/Output : {config_dir}")
    print(f" Scenari Random base   : {num_samples}")
    print(f" Carico Max per f()    : {max_req} req/s")
    print(f"=======================================================\n")

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

    models_to_simulate = []
    for i in range(total_samples):
        m = copy.deepcopy(base_model)
        m.arv_rates = X_synthetic[i, :].tolist()
        models_to_simulate.append(m)

    print(f"Avvio simulazione Go in corso (potrebbe richiedere del tempo)...")
    results = go_simulate(models_to_simulate, n_arrivals=100000, seed=SEED, parallelism=1)

    if not results:
        print("Errore: Nessun risultato ritornato dal simulatore Go.")
        return

    RT = np.zeros((total_samples, num_funcs))
    U = np.zeros((total_samples, num_funcs))
    Cold = np.zeros((total_samples, num_funcs))

    for i, res in enumerate(results):
        RT[i, :] = res["AvgRT"]
        U[i, :] = res["Utility"]
        completions = np.array(res["Completions"])
        cold_starts = np.array(res["ColdStarts"])

        with np.errstate(divide='ignore', invalid='ignore'):
            cold_ratio = cold_starts / completions
            cold_ratio[completions == 0] = 0
        Cold[i, :] = cold_ratio

    max_req_str = str(max_req).replace(".", "_")

    npz_filename = f"dataset_{max_req_str}_{num_samples}.npz"
    pkl_filename = f"base_model_{max_req_str}_{num_samples}.pkl"

    output_npz = os.path.join(config_dir, npz_filename)
    output_pkl = os.path.join(config_dir, pkl_filename)

    np.savez(output_npz, X=X_synthetic, RT=RT, U=U, Cold=Cold)

    with open(output_pkl, "wb") as f:
        pickle.dump(base_model, f)

    print(f"\n[OK] Simulazione completata con successo!")
    print(f"Dataset salvato in: {output_npz}")
    print(f"RT Medio globale: {np.mean(RT):.4f} s")
    print(f"Utility Media globale: {np.mean(U):.4f}")
    print(f"Cold Start % Media globale: {np.mean(Cold) * 100:.2f}%")


if __name__ == "__main__":
    main()