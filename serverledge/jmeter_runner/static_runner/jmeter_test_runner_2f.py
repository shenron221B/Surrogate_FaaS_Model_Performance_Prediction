import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import time
from load_matrix import MATRICES

BASE_DIR = Path("/root/tesi_project/serverledge")
BASE_RESULTS_DIR = BASE_DIR / "results" / "2f_3GBpm" / "q_len_5" / "del"
JMX_PATH = str(BASE_DIR / "2f_jmeter_test.jmx")
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
    time.sleep(2)

    print(f"[CLEAN STATE] Riavvio Serverledge in background...")
    start_cmd = f"{SSH_CMD} '{START_CMD}'"
    subprocess.Popen(start_cmd, shell=True)

    print("[CLEAN STATE] Attesa inizializzazione Serverledge (5 secondi)...\n")
    time.sleep(5)


def print_usage():
    print("Uso corretto dello script:")
    print("  Manuale: python3 jmeter_test_runner_2f.py manual <nome_matrice>")
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print_usage()

    mode = sys.argv[1].lower()

    if mode == "manual":
        matrix_name = sys.argv[2]
        if matrix_name not in MATRICES:
            print(f"Errore: Matrice '{matrix_name}' non trovata in load_matrix.py.")
            sys.exit(1)
        X = MATRICES[matrix_name]
        folder_name = matrix_name
    else:
        print_usage()

    if X.shape[1] < 2:
        print("Errore: La matrice selezionata ha una sola colonna! Serve una matrice N x 2 per questo test.")
        sys.exit(1)

    if X.shape[1] > 2:
        X = X[:, :2]

    results_dir = BASE_RESULTS_DIR / folder_name
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" AVVIO TEST 2 FUNZIONI: {matrix_name.upper()} (Clean State)")
    print(f" Cartella output: {results_dir}")
    print(f" Dimensioni Matrice: {X.shape}")
    print(f"=======================================================\n")

    np.savetxt(os.path.join(results_dir, "X_matrix_used.txt"), X, fmt='%.3f')
    n_rows = X.shape[0]

    for i in range(n_rows):
        restart_serverledge()

        row_dir = os.path.join(results_dir, f"row_{i + 1}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result_{i + 1}.jtl")

        rate_f1 = X[i, 0]
        rate_f2 = X[i, 1]

        cmd = [
            "/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", JMX_PATH,
            "-l", result_file,
            f"-Jseed={SEED}",
            f"-Jrate_f1={rate_f1}",
            f"-Jrate_f2={rate_f2}",
            f"-Jresult_dir={row_dir}"
        ]

        print(f"--- Esecuzione Riga {i + 1}/{n_rows} --- Carico F1: [{rate_f1}] req/s | Carico F2: [{rate_f2}] req/s")
        try:
            subprocess.run(cmd, check=True)
            print("Completato. Preparazione alla riga successiva...")
        except subprocess.CalledProcessError as e:
            print(f"ERRORE JMeter: {e}")

    print("\n[OK] Esecuzioni JMeter completate!")
    print(f"\n[INFO] Avvio elaborazione dataset per 2 funzioni...")

    create_script = str(BASE_DIR / "create_dataset_2f.py")
    try:
        subprocess.run(["python3", create_script, str(results_dir)], check=True)
        print("\n[SUCCESSO] Intera pipeline completata perfettamente!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE] Fallimento durante create_dataset_2f.py: {e}")


if __name__ == "__main__":
    main()