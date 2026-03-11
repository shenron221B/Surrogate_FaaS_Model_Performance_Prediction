import sys
import os
import subprocess
import glob
import json
import pandas as pd
import time
import numpy as np
from pathlib import Path

# import dell'algoritmo smart load explorer
sys.path.append(os.getcwd())
from smart_load_explorer import SmartLoadExplorer

BASE_DIR = Path("/root/tesi_project/serverledge")
SEED = 42

VM1_IP = "192.168.122.4"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."
START_CMD = "/root/serverledge/start_sl.sh"
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"

def print_usage():
    print("\n=========================================================================")
    print(" USO DELLO SCRIPT JMETER SMART RUNNER (19 Argomenti totali)")
    print("=========================================================================")
    print(" Parametri Configurazione Cartella (1-3):")
    print("   1) <Memoria Tot Serverledge MB> (es. 4096)")
    print("   2) <Memoria per singola Func MB> (es. 256)")
    print("   3) <Lunghezza Coda> (es. 0 o 5)")
    print(" Parametri Algoritmo (4-18):")
    print("   4) <Numero Funzioni> (es. 1, 2 o 3)")
    print("   5) <Carico Minimo Partenza> (es. 0.1)")
    print("   6) <Step Size Iniziale> (es. 2.0)")
    print("   7) <Incremento % Step Iniziale> (es. 0.1 per +10%)")
    print("   8) <Metrica Decadimento> (drop_rate_only, utility_only, drop_rate_or_utility, drop_rate_and_utility)")
    print("   9) <Soglia Drop Rate Decadimento> (es. 0.05. Usa 1.0 se utility_only)")
    print("  10) <Soglia Utility Decadimento> (es. 0.85. Usa 1.0 se drop_rate_only)")
    print("  11) <Step Size Critico Iniziale> (es. 0.5)")
    print("  12) <Perc. Accelerazione Step Critico> (es. 0.2 per +20%)")
    print("  13) <Perc. Rallentamento Step Critico> (es. 0.5 per -50%)")
    print("  14) <Delta Lower Bound> (es. 0.02. Sotto questo accelera)")
    print("  15) <Delta Upper Bound> (es. 0.10. Sopra questo rallenta)")
    print("  16) <Metrica Stop> (stesse opzioni di 8)")
    print("  17) <Limite Drop Rate Stop> (es. 0.7. Usa 1.0 se ininfluente)")
    print("  18) <Limite Utility Stop> (es. 0.1. Usa 1.0 se ininfluente)")
    sys.exit(1)

def restart_serverledge():
    print(f"\n[CLEAN STATE] Riavvio ambiente su {VM1_IP}...")
    subprocess.run(f"{SSH_CMD} 'pkill -SIGINT -f serverledge'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(4)
    subprocess.run(f"{SSH_CMD} 'pkill -9 -f serverledge'", shell=True, stderr=subprocess.DEVNULL)
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(10)
    subprocess.Popen(f"{SSH_CMD} '{START_CMD}'", shell=True)
    time.sleep(5)

def evaluate_row(row_dir):
    jtl_files = glob.glob(os.path.join(row_dir, "*.jtl"))
    # lettura dei ddati della funzione per media
    txt_files = glob.glob(os.path.join(row_dir, "http_responses_func1.txt"))

    jm_tot, jm_fail, jm_utility = 0, 0, 0.0
    sl_dur_mean = 0.205

    if txt_files:
        dur = []
        try:
            with open(txt_files[0], 'r') as f:
                for line in f:
                    if line.startswith("{"):
                        d = json.loads(line)
                        if d.get("Success"): dur.append(d.get("Duration", 0))
            if dur: sl_dur_mean = np.mean(dur)
        except:
            pass

    if jtl_files:
        try:
            df = pd.read_csv(jtl_files[0])
            df['rt_sec'] = df['elapsed'] / 1000.0
            jm_tot = len(df)
            if jm_tot > 0:
                jm_fail = len(df[df['success'] == False])
                valid = df[(df['success'] == True) & (df['rt_sec'] < (sl_dur_mean * 2.5))]
                jm_utility = len(valid) / jm_tot
        except:
            pass

    fail_rate = (jm_fail / jm_tot) if jm_tot > 0 else 1.0
    return jm_utility, fail_rate

def main():
    if len(sys.argv) != 19:
        print_usage()

    # parsing e costruzione cartella
    tot_mem = int(sys.argv[1])
    func_mem = int(sys.argv[2])
    q_len = int(sys.argv[3])
    num_funcs = int(sys.argv[4])

    # seleziona dinamicamente il JMX corretto
    jmx_path = str(BASE_DIR / f"{num_funcs}f_jmeter_test.jmx")
    if not os.path.exists(jmx_path):
        print(f"[ERRORE] File JMX non trovato: {jmx_path}")
        sys.exit(1)

    gb_mem = tot_mem // 1024
    folder_path = BASE_DIR / "results" / "sl_restart" / "smart_explorer" / f"{num_funcs}f_{gb_mem}GBpm" / f"q_len_{q_len}" / str(func_mem)
    os.makedirs(folder_path, exist_ok=True)
    
    # inizializzazione dell'explorer
    explorer = SmartLoadExplorer(
        num_funcs=num_funcs, min_load=float(sys.argv[5]),
        initial_step_size=float(sys.argv[6]), initial_step_inc_perc=float(sys.argv[7]),
        decay_metric=sys.argv[8], decay_drop_thresh=float(sys.argv[9]), decay_util_thresh=float(sys.argv[10]),
        critical_step_size=float(sys.argv[11]), crit_step_accel_perc=float(sys.argv[12]),
        crit_step_decel_perc=float(sys.argv[13]), crit_diff_lower_bound=float(sys.argv[14]),
        crit_diff_upper_bound=float(sys.argv[15]), stop_metric=sys.argv[16],
        stop_drop_limit=float(sys.argv[17]), stop_util_limit=float(sys.argv[18])
    )

    print(f"\n{'=' * 75}")
    print(f" AVVIO JMETER SMART RUNNER ({num_funcs} FUNZIONI)")
    print(f" Output : {folder_path}")
    print(f"{'=' * 75}\n")

    row_index = 1
    matrix_used = []

    while True:
        load_vector = explorer.get_next_load()
        if load_vector is None:
            break

        matrix_used.append(load_vector)
        restart_serverledge()

        row_dir = os.path.join(folder_path, f"row_{row_index}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result_{row_index}.jtl")

        # costruzione dinamica della stringa di comando per N funzioni
        cmd = ["/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", jmx_path,
               "-l", result_file, f"-Jseed={SEED}", f"-Jresult_dir={row_dir}"]

        for i in range(num_funcs):
            cmd.append(f"-Jrate_f{i + 1}={load_vector[i]}")

        print(f"\n--- Riga {row_index} --- Carico richiesto: {load_vector} req/s")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)

        utility, fail_rate = evaluate_row(row_dir)
        explorer.report_result(load_vector, utility, fail_rate)

        row_index += 1

    np.savetxt(os.path.join(folder_path, "X_matrix_used.txt"), np.array(matrix_used), fmt='%.3f')

    print("\n[OK] Esplorazione Conclusa. Generazione Dataset...")
    create_script = str(BASE_DIR / f"create_dataset_{num_funcs}f.py")
    if os.path.exists(create_script):
        subprocess.run(["python3", create_script, str(folder_path)], check=True)
        print("\n[SUCCESSO] Pipeline completata!")
    else:
        print(f"\n[ATTENZIONE] Script {create_script} non trovato. Elaborazione dataset saltata.")


if __name__ == "__main__":
    main()