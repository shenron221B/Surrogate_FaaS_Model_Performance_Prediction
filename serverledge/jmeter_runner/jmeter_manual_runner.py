import numpy as np
import os
import subprocess
import sys
from pathlib import Path
import time

BASE_DIR = Path("/root/tesi_project/serverledge")
sys.path.append(str(BASE_DIR / "utility"))
try:
    from load_matrix import MATRICES
except ImportError:
    print("[ERRORE] File 'load_matrix.py' non trovato in utility/.")
    sys.exit(1)

SEED = 42

# configurazione accesso remoto
VM1_IP = "192.168.122.6"
VM1_PORT = "22"
VM1_PASSWORD = "Farag01."

# comandi systemd
START_CMD = "systemctl start serverledge"
STOP_CMD = "systemctl stop serverledge"
SSH_CMD = f"sshpass -p '{VM1_PASSWORD}' ssh -p {VM1_PORT} -o StrictHostKeyChecking=no root@{VM1_IP}"


def restart_serverledge():
    print(f"\n[CLEAN STATE] Inizio procedura di pulizia remota su {VM1_IP}...")

    # spegnimento di Serverledge
    subprocess.run(f"{SSH_CMD} '{STOP_CMD}'", shell=True, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # pulizia container
    print(f"[CLEAN STATE] Rimozione container funzione (warm/zombie)...")
    clean_docker_cmd = "docker ps -a --format '{{.Names}}' | grep -vE '^(etcd-server|minio-server)$' | xargs -r docker rm -f"
    subprocess.run(f"{SSH_CMD} \"{clean_docker_cmd}\"", shell=True, stderr=subprocess.DEVNULL)

    # riavvio di etcd
    print(f"[CLEAN STATE] Riavvio di etcd-server...")
    subprocess.run(f"{SSH_CMD} 'docker stop etcd-server && docker start etcd-server'", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)

    # riavvio Serverledge
    print(f"[CLEAN STATE] Riavvio Serverledge in background...")
    subprocess.run(f"{SSH_CMD} '{START_CMD}'", shell=True)
    print("[CLEAN STATE] Attesa inizializzazione Serverledge (5 secondi)...\n")
    time.sleep(5)


def print_usage():
    print("\n================================================================")
    print(" USO DELLO SCRIPT JMETER MANUAL RUNNER")
    print("==================================================================")
    print(" Argomenti:")
    print("   1) <Numero Funzioni> (es. 1, 2, 3, 4)")
    print("   2) <Nome Matrice> (es. weather_short, eval_4f_mix)")
    print("   3) <Path file JMX> (es. /root/.../4f_jmeter_test.jmx)")
    print("   4) <Directory Output Base> (es. /root/.../manual_explorer/4f_mix)")
    print("\n Esempio 1F:")
    print("   python3 jmeter_manual_runner.py 1 weather_short /path/weather.jmx /output/1f")
    print(" Esempio 4F:")
    print("   python3 jmeter_manual_runner.py 4 matrix_4f_high /path/4f_test.jmx /output/4f\n")
    sys.exit(1)


def main():
    if len(sys.argv) != 5:
        print_usage()

    try:
        num_funcs = int(sys.argv[1])
    except ValueError:
        print("[ERRORE] Il numero di funzioni deve essere un intero.")
        sys.exit(1)

    matrix_name = sys.argv[2]
    jmx_path_arg = sys.argv[3]
    base_results_arg = sys.argv[4]

    # recupero matrice
    if matrix_name not in MATRICES:
        print(f"[ERRORE] Matrice '{matrix_name}' non trovata in load_matrix.py.")
        sys.exit(1)

    X = MATRICES[matrix_name]

    # validazione matrice
    if X.shape[1] < num_funcs:
        print(
            f"[ERRORE] La matrice '{matrix_name}' ha solo {X.shape[1]} colonne, ma hai richiesto {num_funcs} funzioni!")
        sys.exit(1)

    # taglia le colonne in base al numero di funzioni
    X = X[:, :num_funcs]

    # validazione JMX
    JMX_PATH = os.path.abspath(jmx_path_arg)
    if not os.path.exists(JMX_PATH):
        print(f"[ERRORE] File JMX '{JMX_PATH}' non trovato.")
        sys.exit(1)

    # setup directory
    BASE_RESULTS_DIR = Path(os.path.abspath(base_results_arg))
    results_dir = BASE_RESULTS_DIR / matrix_name
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f" AVVIO TEST {num_funcs} FUNZIONI (Manuale - Matrice: {matrix_name})")
    print(f" Target JMX : {JMX_PATH}")
    print(f" Output Dir : {results_dir}")
    print(f" Dims Matrice: {X.shape}")
    print(f"{'=' * 70}\n")

    np.savetxt(os.path.join(results_dir, "X_matrix_used.txt"), X, fmt='%.3f')
    n_rows = X.shape[0]

    for i in range(n_rows):
        restart_serverledge()

        row_dir = os.path.join(results_dir, f"row_{i + 1}")
        os.makedirs(row_dir, exist_ok=True)
        result_file = os.path.join(row_dir, f"result_{i + 1}.jtl")

        # costruzione dell'array dei rate dinamica
        rates = X[i]
        rate_params = [f"-Jrate_f{j + 1}={rates[j]}" for j in range(num_funcs)]

        cmd = [
                  "/root/apache-jmeter-5.6.3/bin/jmeter", "-n", "-t", JMX_PATH,
                  "-l", result_file, f"-Jseed={SEED}", f"-Jresult_dir={row_dir}"
              ] + rate_params

        rates_str = " | ".join([f"F{j + 1}: {rates[j]} req/s" for j in range(num_funcs)])
        print(f"--- Esecuzione Riga {i + 1}/{n_rows} --- Carichi: [ {rates_str} ]")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print("Completato. Preparazione alla riga successiva...")
        except subprocess.CalledProcessError as e:
            print(f"[ERRORE] JMeter ha fallito sulla riga {i + 1}: {e}")

    print("\n[OK] Esecuzioni JMeter completate!")
    print(f"\n[INFO] Avvio elaborazione dataset per {num_funcs} funzioni...")

    create_script = str(BASE_DIR / "utility" / "create_dataset_2.py")
    if os.path.exists(create_script):
        try:
            subprocess.run(["python3", create_script, str(results_dir), str(num_funcs)], check=True)
            print("\n[SUCCESSO] Intera pipeline completata perfettamente!")
        except subprocess.CalledProcessError as e:
            print(f"\n[ERRORE] Fallimento durante la creazione del dataset: {e}")
    else:
        print(f"\n[ATTENZIONE] Script {create_script} non trovato! Elaborazione saltata.")

    print("\n[INFO] Esecuzione pulizia finale dell'ambiente...")
    restart_serverledge()
    print("[OK] Ambiente pulito e pronto per il prossimo test.")

if __name__ == "__main__":
    main()