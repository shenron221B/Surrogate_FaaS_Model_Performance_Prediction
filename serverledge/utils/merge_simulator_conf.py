import sys
import os
import glob
import yaml
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 merge_simulator_conf.py <path_directory_principale>")
        print("Es: python3 merge_simulator_conf.py /root/.../poisson_600")
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])

    # cerca simulator-conf.yml in tutte le sottocartelle immediate
    yml_files = glob.glob(os.path.join(root_dir, "*", "simulator-conf.yml"))

    if not yml_files:
        print(f"[ERRORE] Nessun simulator-conf.yml trovato nelle sottocartelle di {root_dir}")
        sys.exit(1)

    print(f"\n--- MERGE SIMULATOR CONFIG ---")
    print(f"Trovati {len(yml_files)} file YAML da mediare.")

    base_conf = None
    net_overheads = []
    serv_time_durations = []
    serv_time_cvs = []
    init_times = []

    for f in yml_files:
        with open(f, 'r') as file:
            conf = yaml.safe_load(file)

            # usa il primo file trovato come "scheletro" per le costanti (functions, mem_demands, ecc.)
            if base_conf is None:
                base_conf = conf.copy()

            # estrae i valori numerici da mediare
            net_overheads.append(conf.get('net_overhead', 0.0))
            serv_time_durations.append(conf.get('serv_time_duration', []))
            serv_time_cvs.append(conf.get('serv_time_cvs', []))
            init_times.append(conf.get('init_times', []))

    # calcolo delle medie vettoriali (axis=0 calcola la media colonna per colonna, cioè funzione per funzione)
    base_conf['net_overhead'] = float(np.mean(net_overheads))
    base_conf['serv_time_duration'] = np.mean(serv_time_durations, axis=0).tolist()
    base_conf['serv_time_cvs'] = np.mean(serv_time_cvs, axis=0).tolist()
    base_conf['init_times'] = np.nanmean(init_times, axis=0).tolist()

    # salvataggio del master YAML nella cartella radice
    out_path = os.path.join(root_dir, "simulator-conf.yml")
    with open(out_path, 'w') as out_f:
        yaml.dump(base_conf, out_f, sort_keys=False)

    print(f"[OK] Master YAML calcolato e salvato in:\n -> {out_path}")


if __name__ == "__main__":
    main()