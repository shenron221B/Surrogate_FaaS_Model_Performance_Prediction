import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import time
from load_matrix import MATRICES, generate_random_matrix

BASE_DIR = Path("/root/tesi_project/serverledge")
BASE_RESULTS_DIR = BASE_DIR / "results" / "1f_2GBpm" / "del" / "q_len_5"
JMX_PATH = str(BASE_DIR / "1f_jmeter_test.jmx")
SEED = 42

# configurazione accesso remoto
VM1_IP = "192.168.122.4"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."

# comando di avvio di Serverledge
SERVERLEDGE_PATH = "/root/serverledge"
START_CMD = "/root/serverledge/start_sl.sh"

# prefisso universale per lanciare comandi via SSH con password in automatico
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"

def restart_serverledge():
    print(f"\n[CLEAN STATE] Inizio procedura di pulizia remota su {VM1_IP}...")

    # spegnimento Ctrl+C remoto
    print(f"[CLEAN STATE] Chiusura di Serverledge...")
    subprocess.run(f"{SSH_CMD} 'pkill -SIGINT -f serverledge'", shell=True, stderr=subprocess.DEVNULL)

    # 4 secondi per la chiusura pulita
    time.sleep(4)

    # nel caso Serverledge fosse bloccato
    subprocess.run(f"{SSH_CMD} 'pkill -9 -f serverledge'", shell=True, stderr=subprocess.DEVNULL)

    # riavvio di etcd remoto
    print(f"[CLEAN STATE] Riavvio di etcd-server...")
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # riavvio di Serverledge remoto in background
    print(f"[CLEAN STATE] Riavvio Serverledge in background...")
    # questo comando entra nella cartella su VM1 e lancia Serverledge slegandolo dal terminale SSH
    start_cmd = f"{SSH_CMD} '{START_CMD}'"
    subprocess.Popen(start_cmd, shell=True)

    # attesa che il server sia pronto ad accettare chiamate HTTP
    print("[CLEAN STATE] Attesa inizializzazione Serverledge (5 secondi)...\n")
    time.sleep(5)


def print_usage():
    print("uso corretto dello script:")
    print("  Manuale: python3 jmeter_test_runner_1f.py manual <nome_matrice>")
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
    else:
        print_usage()

    if X.shape[1] > 1:
        X = X[:, :1]

    results_dir = BASE_RESULTS_DIR / folder_name
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" AVVIO TEST 1 FUNZIONE: {matrix_name.upper()} (Coda Corta & Clean State)")
    print(f" Cartella output: {results_dir}")
    print(f" Dimensioni Matrice: {X.shape}")
    print(f"=======================================================\n")

    np.savetxt(os.path.join(results_dir, "X_matrix_used.txt"), X, fmt='%.3f')
    n_rows = X.shape[0]

    for i in range(n_rows):
        # riavvio remoto di Serverledge
        restart_serverledge()

        row_dir = os.path.join(results_dir, f"row_{i + 1}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result_{i + 1}.jtl")

        rate = X[i, 0]
        cmd = [
            "/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", JMX_PATH,
            "-l", result_file, f"-Jseed={SEED}", f"-Jrate_f1={rate}", f"-Jresult_dir={row_dir}"
        ]

        print(f"--- Esecuzione Riga {i + 1}/{n_rows} --- Carico: [{rate}] req/s")
        try:
            subprocess.run(cmd, check=True)
            print("Completato. Preparazione alla riga successiva...")
        except subprocess.CalledProcessError as e:
            print(f"ERRORE JMeter: {e}")

    print("\n[OK] Esecuzioni JMeter completate!")
    print(f"\n[INFO] Avvio elaborazione dataset...")

    # chiamata al dataset creation
    create_script = str(BASE_DIR / "create_dataset_1f.py")
    try:
        subprocess.run(["python3", create_script, str(results_dir)], check=True)
        print("\n[SUCCESSO] Intera pipeline completata perfettamente!")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERRORE] Fallimento durante create_dataset_1f.py: {e}")


if __name__ == "__main__":
    main()