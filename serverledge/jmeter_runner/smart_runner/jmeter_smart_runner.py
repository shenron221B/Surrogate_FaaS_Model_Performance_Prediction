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
BASE_DIR = Path("/root/tesi_project/serverledge")
SEED = 42

# import dell'algoritmo smart load explorer dalla cartella utility
sys.path.append(str(BASE_DIR / "utility"))
try:
    from smart_load_explorer import SmartLoadExplorer
except ImportError:
    print("[ERRORE] File 'smart_load_explorer.py' non trovato in utility/.")
    sys.exit(1)

# configurazione accesso remoto
VM1_IP = "192.168.122.6"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."

# Comandi Systemd per VM3
START_CMD = "systemctl start serverledge"
STOP_CMD = "systemctl stop serverledge"
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"


def print_usage():
    print("\n=========================================================================")
    print(" USO DELLO SCRIPT JMETER SMART RUNNER UNIFIED (17 Argomenti totali)")
    print("=========================================================================")
    print(" Parametri Configurazione Path (1-3):")
    print("   1) <Path file .jmx> (es. /root/.../weather_jmeter_test.jmx)")
    print("   2) <Directory Output Base> (es. /root/.../smart_explorer/1f_weather_4GBpm/q_len_5/256)")
    print("   3) <Numero Funzioni> (es. 1, 2 o 3)")
    print(" Parametri Algoritmo Explorer (4-17):")
    print("   4) <Carico Minimo Partenza> (es. 10.0)")
    print("   5) <Step Size Iniziale> (es. 5.0)")
    print("   6) <Incremento % Step Iniziale> (es. 0.0)")
    print("   7) <Metrica Decadimento> (drop_rate_only, utility_only, drop_rate_or_utility, drop_rate_and_utility)")
    print("   8) <Soglia Drop Rate Decadimento> (es. 1.0 se utility_only)")
    print("   9) <Soglia Utility Decadimento> (es. 0.90)")
    print("  10) <Step Size Critico Iniziale> (es. 1.0)")
    print("  11) <Perc. Accelerazione Step Critico> (es. 0.0)")
    print("  12) <Perc. Rallentamento Step Critico> (es. 0.0)")
    print("  13) <Delta Lower Bound> (es. 0.05)")
    print("  14) <Delta Upper Bound> (es. 0.10)")
    print("  15) <Metrica Stop> (stesse opzioni di 7)")
    print("  16) <Limite Drop Rate Stop> (es. 0.60)")
    print("  17) <Limite Utility Stop> (es. 1.0 se ininfluente)")
    sys.exit(1)


def restart_serverledge():
    print(f"\n[CLEAN STATE] Riavvio ambiente su {VM1_IP} tramite systemd...")

    # spegnimento di Serverledge
    subprocess.run(f"{SSH_CMD} '{STOP_CMD}'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # eliminazione dei container etcd-server e minio-server
    print(f"[CLEAN STATE] Rimozione container funzione (warm/zombie)...")
    clean_docker_cmd = "docker ps -a --format '{{.Names}}' | grep -vE '^(etcd-server|minio-server)$' | xargs -r docker rm -f"
    subprocess.run(f"{SSH_CMD} \"{clean_docker_cmd}\"", shell=True, stderr=subprocess.DEVNULL)

    # riavvio di Etcd
    print(f"[CLEAN STATE] Riavvio etcd-server...")
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # riavvio di Serverledge
    print(f"[CLEAN STATE] Avvio Serverledge...")
    subprocess.run(f"{SSH_CMD} '{START_CMD}'", shell=True)
    time.sleep(5)


def evaluate_row(row_dir):
    jtl_files = glob.glob(os.path.join(row_dir, "*.jtl"))
    # lettura dei dati della funzione per media (fissato a func1 per ora)
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
    if len(sys.argv) != 18:
        print_usage()

    # Parsing argomenti base
    jmx_path = os.path.abspath(sys.argv[1])
    folder_path = Path(os.path.abspath(sys.argv[2]))
    num_funcs = int(sys.argv[3])

    if not os.path.exists(jmx_path):
        print(f"[ERRORE] File JMX non trovato: {jmx_path}")
        sys.exit(1)

    os.makedirs(folder_path, exist_ok=True)

    # inizializzazione dell'explorer
    explorer = SmartLoadExplorer(
        num_funcs=num_funcs, min_load=float(sys.argv[4]),
        initial_step_size=float(sys.argv[5]), initial_step_inc_perc=float(sys.argv[6]),
        decay_metric=sys.argv[7], decay_drop_thresh=float(sys.argv[8]), decay_util_thresh=float(sys.argv[9]),
        critical_step_size=float(sys.argv[10]), crit_step_accel_perc=float(sys.argv[11]),
        crit_step_decel_perc=float(sys.argv[12]), crit_diff_lower_bound=float(sys.argv[13]),
        crit_diff_upper_bound=float(sys.argv[14]), stop_metric=sys.argv[15],
        stop_drop_limit=float(sys.argv[16]), stop_util_limit=float(sys.argv[17])
    )

    print(f"\n{'=' * 75}")
    print(f" AVVIO JMETER SMART RUNNER ({num_funcs} FUNZIONI)")
    print(f" JMX Target : {jmx_path}")
    print(f" Output Dir : {folder_path}")
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
    # Fix: Cerca lo script di dataset in utility
    create_script = str(BASE_DIR / "utility" / f"create_dataset_{num_funcs}f.py")
    if os.path.exists(create_script):
        subprocess.run(["python3", create_script, str(folder_path)], check=True)
        print("\n[SUCCESSO] Pipeline completata!")
    else:
        print(f"\n[ATTENZIONE] Script {create_script} non trovato. Elaborazione dataset saltata.")


if __name__ == "__main__":
    main()