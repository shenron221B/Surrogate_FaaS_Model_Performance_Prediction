import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import time
from load_matrix import MATRICES, generate_random_matrix

BASE_DIR = Path("/root/tesi_project/serverledge")
BASE_RESULTS_DIR = BASE_DIR / "results" / "3f_3GBpm" / "q_len_10" / "del"
JMX_PATH = str(BASE_DIR / "3f_jmeter_test.jmx")
SEED = 42

VM1_IP = "192.168.122.4"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."

START_CMD = "/root/serverledge/start_sl.sh"
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"


def restart_serverledge():
    print(f"\n[CLEAN STATE] Inizio procedura di pulizia remota su {VM1_IP}...")

    print(f"[CLEAN STATE] Chiusura di Serverledge...")
    subprocess.run(f"{SSH_CMD} 'pkill -SIGINT -f serverledge'", shell=True, stderr=subprocess.DEVNULL)

    time.sleep(4)
    subprocess.run(f"{SSH_CMD} 'pkill -9 -f serverledge'", shell=True, stderr=subprocess.DEVNULL)

    print(f"[CLEAN STATE] Riavvio di etcd-server...")
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print("[CLEAN STATE] Attesa inizializzazione Etcd (10 secondi)...")
    time.sleep(10)

    print(f"[CLEAN STATE] Riavvio Serverledge in background...")
    start_cmd = f"{SSH_CMD} '{START_CMD}'"
    subprocess.Popen(start_cmd, shell=True)

    print("[CLEAN STATE] Attesa inizializzazione Serverledge (5 secondi)...\n")
    time.sleep(5)


def print_usage():
    print("uso corretto dello script:")
    print("  Manuale: python3 jmeter_test_runner_3f.py manual <nome_matrice>")
    print("  Random:  python3 jmeter_test_runner_3f.py random <max_load> <n_righe>")
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print_usage()

    mode = sys.argv[1].lower()

    if mode == "manual":
        matrix_name = sys.argv[2]
        if matrix_name not in MATRICES:
            print(f"Errore: Matrice '{matrix_name}' non trovata.")
            sys.exit(1)
        X = MATRICES[matrix_name]
        folder_name = matrix_name
    elif mode == "random":
        if len(sys.argv) < 4:
            print_usage()
        try:
            max_load = float(sys.argv[2])
            n_rows = int(sys.argv[3])
        except ValueError:
            print("Errore argomenti random.")
            sys.exit(1)
        X = generate_random_matrix(max_load, n_rows)
        load_str = str(max_load).replace(".", "_")
        folder_name = f"r_x_{load_str}_c{n_rows}"
    else:
        print_usage()

    if X.shape[1] < 3:
        print("Errore: La matrice selezionata ha meno di 3 colonne! Serve una matrice N x 3 per questo test.")
        sys.exit(1)

    X = X[:, :3]

    results_dir = BASE_RESULTS_DIR / folder_name
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" AVVIO TEST 3 FUNZIONI: Modalità {mode.upper()} (Clean State)")
    print(f" Cartella output: {results_dir}")
    print(f" Dimensioni Matrice adattata: {X.shape}")
    print(f"=======================================================\n")

    np.savetxt(os.path.join(results_dir, "X_matrix_used.txt"), X, fmt='%.3f')

    n_rows, n_cols = X.shape

    for i in range(n_rows):
        restart_serverledge()

        row_dir = os.path.join(results_dir, f"row_{i + 1}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result_{i + 1}.jtl")

        rate_params = [f"-Jrate_f{j + 1}={X[i, j]}" for j in range(n_cols)]

        cmd = [
            "/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", JMX_PATH,
            "-l", result_file, f"-Jseed={SEED}", *rate_params, f"-Jresult_dir={row_dir}"
        ]

        print(f"--- Esecuzione Riga {i + 1}/{n_rows} --- Carichi: {np.round(X[i], 3)}")
        try:
            subprocess.run(cmd, check=True)
            print("Completato. Preparazione alla riga successiva...")
        except subprocess.CalledProcessError as e:
            print(f"ERRORE JMeter: {e}")

    print("\n[OK] Esecuzioni JMeter completate!")
    print(f"\n[INFO] Avvio elaborazione dataset e configurazione simulatore...")
    create_script = str(BASE_DIR / "create_dataset_3f.py")

    try:
        subprocess.run(["python3", create_script, str(results_dir)], check=True)
        print("\n[SUCCESSO] Intera pipeline completata perfettamente!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE] Fallimento durante create_dataset_3f.py: {e}")


if __name__ == "__main__":
    main()