import sys
import os
import subprocess
import glob
import json
import pandas as pd
import time
import numpy as np
from pathlib import Path

BASE_DIR = Path("/root/tesi_project/serverledge_results")
SEED = 42

sys.path.append(str(BASE_DIR / "utils"))
try:
    from nf_test_configuration import CONFIG_MATRICES
    from smart_load_explorer_nf import SmartLoadExplorerNF
except ImportError as e:
    print(f"[ERRORE] File di utils non trovati: {e}")
    sys.exit(1)

VM1_IP = "192.168.122.6"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."
START_CMD = "systemctl start serverledge_results"
STOP_CMD = "systemctl stop serverledge_results"
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"


def print_usage():
    print("\n=========================================================================")
    print(" USO: python3 jmeter_smart_runner.py <N> <Pool_Mem> <Q_Len> <Mem_Req> <JMX_Path> <Config_Name>")
    print(" Esempio: python3 jmeter_smart_runner.py 3 3 5 256 /path/3f_hash_matrix_pi.jmx test_3f_mix")
    print("=========================================================================\n")
    sys.exit(1)


def restart_serverledge():
    print(f"\n[CLEAN STATE] Riavvio ambiente su {VM1_IP}...")
    subprocess.run(f"{SSH_CMD} '{STOP_CMD}'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2)
    clean_cmd = "docker ps -a --format '{{.Names}}' | grep -vE '^(etcd-server|minio-server)$' | xargs -r docker rm -f"
    subprocess.run(f"{SSH_CMD} \"{clean_cmd}\"", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    subprocess.run(f"{SSH_CMD} '{START_CMD}'", shell=True)
    print("[CLEAN STATE] Attesa inizializzazione Serverledge (5 secondi)...")
    time.sleep(5)


def evaluate_nf_row(base_row_dir, num_funcs, func_names):
    utilities, fail_rates = [], []
    jm_tots, jm_fails = [], []

    jtl_files = glob.glob(os.path.join(base_row_dir, "result.jtl"))
    df_all = None
    if jtl_files:
        try:
            df_all = pd.read_csv(jtl_files[0])
            df_all['rt_sec'] = df_all['elapsed'] / 1000.0
        except:
            pass

    # dizionario con i tempi medi pre-calcolati per evitare letture pesanti dal JSON
    base_durations = {
        "weather": 0.022,
        "pi_calc": 0.314,
        "hash_worker": 0.131,
        "matrix_mem": 0.097
    }

    # valutazione funzione separatamente
    for i in range(num_funcs):
        func_id = i + 1
        f_name = func_names[i]
        label = f'Invoke_func_{func_id}'

        jm_tot, jm_fail, jm_utility = 0, 0, 0.0

        sl_dur_mean = base_durations.get(f_name, 0.200)

        utility_threshold = sl_dur_mean * 2.5

        # filtraggio del JTL solo per la funzione corrente
        if df_all is not None:
            df_func = df_all[df_all['label'] == label]
            jm_tot = len(df_func)
            if jm_tot > 0:
                jm_fail = len(df_func[df_func['success'] == False])
                valid = df_func[(df_func['success'] == True) & (df_func['rt_sec'] < utility_threshold)]
                jm_utility = len(valid) / jm_tot

        fail_rate = (jm_fail / jm_tot) if jm_tot > 0 else 1.0
        utilities.append(jm_utility)
        fail_rates.append(fail_rate)
        jm_tots.append(jm_tot)
        jm_fails.append(jm_fail)

    return utilities, fail_rates, jm_tots, jm_fails


def main():
    if len(sys.argv) != 7:
        print_usage()

    num_funcs = int(sys.argv[1])
    pool_mem = sys.argv[2]
    q_len = sys.argv[3]
    mem_req = sys.argv[4]
    jmx_path = os.path.abspath(sys.argv[5])
    config_name = sys.argv[6]

    if config_name not in CONFIG_MATRICES:
        print(f"[ERRORE] Configurazione '{config_name}' non trovata in nf_test_configuration.py")
        sys.exit(1)

    config = CONFIG_MATRICES[config_name]
    if len(config["func_names"]) != num_funcs:
        print(f"[ERRORE] Configurazione ha {len(config['func_names'])} funzioni, ma N={num_funcs}")
        sys.exit(1)

    # costruzione del path
    funcs_concat = "_".join([f.replace("_", "") for f in config["func_names"]])
    out_dir = BASE_DIR / "results" / "sl_restart" / "smart_explorer" / "real_functions" / f"{num_funcs}f_{pool_mem}GBpm" / f"q_len_{q_len}" / mem_req / funcs_concat
    os.makedirs(out_dir, exist_ok=True)

    explorer = SmartLoadExplorerNF(config)
    row_index = 1
    matrix_used = []

    print(f"\n{'=' * 75}")
    print(f" AVVIO SMART EXPLORER MULTI-FUNZIONE ({num_funcs}F)")
    print(f" Config   : {config_name}")
    print(f" Out Dir  : {out_dir}")
    print(f"{'=' * 75}\n")

    while True:
        load_vector = explorer.get_next_load()
        if load_vector is None:
            break

        matrix_used.append(load_vector)
        restart_serverledge()

        row_dir = os.path.join(out_dir, f"row_{row_index}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result.jtl")

        cmd = ["/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", jmx_path,
               "-l", result_file, f"-Jseed={SEED}", f"-Jresult_dir={row_dir}"]

        for i in range(num_funcs):
            cmd.append(f"-Jrate_f{i + 1}={load_vector[i]}")

        print(f"\n--- Esecuzione Riga {row_index} --- Carichi: {load_vector}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        utilities, fail_rates, jm_tots, jm_fails = evaluate_nf_row(row_dir, num_funcs, config["func_names"])

        print("\n  [Valutazione Prestazioni]")
        for i in range(num_funcs):
            print(
                f"    [F{i + 1}: {config['func_names'][i]}] State: {explorer.states[i]} | Load: {load_vector[i]} req/s | U: {utilities[i]:.4f} | Fail: {fail_rates[i] * 100:.1f}%")

        explorer.report_result(load_vector, utilities, fail_rates)
        row_index += 1

    np.savetxt(os.path.join(out_dir, "X_matrix_used.txt"), np.array(matrix_used), fmt='%.3f')

    print("\n[OK] Esplorazione Globale Conclusa. Generazione dataset...")
    create_script = str(BASE_DIR / "utils" / "create_dataset.py")
    if os.path.exists(create_script):
        subprocess.run(["python3", create_script, str(out_dir), str(num_funcs)], check=True)
        print("\n[SUCCESSO] Dataset Multi-Funzione completato!")
    else:
        print(f"\n[ATTENZIONE] Script {create_script} non trovato! Elaborazione saltata.")

    restart_serverledge()
    print("[OK] Ambiente pulito e terminato.")


if __name__ == "__main__":
    main()
