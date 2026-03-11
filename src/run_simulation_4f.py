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
    print("Uso corretto: python3 run_simulation.py <dir_configurazione> <max_req> <num_scenari>")
    print("Esempio: python3 run_simulation.py /root/tesi_project/serverledge/results/x1 1.5 5000")
    sys.exit(1)


def main():
    # controllo argomenti in input
    if len(sys.argv) < 4:
        print_usage()

    config_dir = sys.argv[1]
    try:
        max_req = float(sys.argv[2])
        num_samples = int(sys.argv[3])
    except ValueError:
        print("Errore: max_req deve essere un numero decimale (es. 1.5) e num_scenari un intero (es. 5000).")
        sys.exit(1)

    conf_file = os.path.join(config_dir, "simulator-conf.yml")

    if not os.path.exists(conf_file):
        print(f"Errore: File di configurazione non trovato in {conf_file}")
        sys.exit(1)

    # caricamento configurazione YAML generata dai dati reali
    with open(conf_file, 'r') as f:
        config_data = yaml.safe_load(f)

    base_model = model_from_conf(config_data)

    print(f"\n=======================================================")
    print(f" AVVIO SIMULAZIONE SINTETICA")
    print(f" Cartella Input/Output : {config_dir}")
    print(f" Scenari da generare   : {num_samples}")
    print(f" Carico Max per f()    : {max_req} req/s")
    print(f"=======================================================\n")

    # generazione matrice di carichi random
    rng = np.random.default_rng(SEED)
    num_funcs = len(base_model.mem_demands)
    X_synthetic = rng.uniform(0.05, max_req, (num_samples, num_funcs))

    models_to_simulate = []
    for i in range(num_samples):
        m = copy.deepcopy(base_model)
        m.arv_rates = X_synthetic[i, :].tolist()
        models_to_simulate.append(m)

    print(f"Avvio simulazione Go...")
    results = go_simulate(models_to_simulate, n_arrivals=100000, seed=SEED, parallelism=1)

    if not results:
        print("Errore: Nessun risultato ritornato dal simulatore Go.")
        return

    RT = np.zeros((num_samples, num_funcs))
    U = np.zeros((num_samples, num_funcs))
    Cold = np.zeros((num_samples, num_funcs))

    for i, res in enumerate(results):
        RT[i, :] = res["AvgRT"]
        U[i, :] = res["Utility"]
        completions = np.array(res["Completions"])
        cold_starts = np.array(res["ColdStarts"])

        with np.errstate(divide='ignore', invalid='ignore'):
            cold_ratio = cold_starts / completions
            cold_ratio[completions == 0] = 0
        Cold[i, :] = cold_ratio

    # costruzione dinamica del nome del file
    max_req_str = str(max_req).replace(".", "_")
    npz_filename = f"dataset_{max_req_str}_{num_samples}.npz"
    pkl_filename = f"base_model_{max_req_str}_{num_samples}.pkl"

    output_npz = os.path.join(config_dir, npz_filename)
    output_pkl = os.path.join(config_dir, pkl_filename)

    # salvataggio dati e modello base
    np.savez(output_npz, X=X_synthetic, RT=RT, U=U, Cold=Cold)

    with open(output_pkl, "wb") as f:
        pickle.dump(base_model, f)

    print(f"\n[OK] Simulazione completata con successo!")
    print(f"Dataset salvato in: {output_npz}")
    print(f"RT Medio: {np.mean(RT):.4f} s")
    print(f"Utility Media: {np.mean(U):.4f}")
    print(f"Cold Start % Media: {np.mean(Cold) * 100:.2f}%")


if __name__ == "__main__":
    main()