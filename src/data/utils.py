# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------


import numpy as np
import tempfile
import subprocess
import json
import shlex
import os
import multiprocessing

from models.model import ModelEncoder

WARNING = False


def run_process_subprocess_run(command):
    """
    Run a process using subprocess.run() - recommended for Python 3.5+
    
    Args:
        command (str): Command to execute
    
    Returns:
        subprocess.CompletedProcess: Process result
    """
    try:
        # Split the command to handle arguments safely
        args = shlex.split(command)
        
        # Run the process and capture output
        result = subprocess.run(
            args, 
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Return strings instead of bytes
            check=True            # Raise CalledProcessError for non-zero exit codes
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Process failed with exit code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return None


# Lancia il simulatore da riga di comando esternamente
def go_simulate(models, n_arrivals=10**6, seed=1, parallelism = None):
    global WARNING
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SIMULATOR_DIR = os.path.join(SCRIPT_DIR, '..', 'simulator_go')
    SIMULATOR = os.path.join(SIMULATOR_DIR, 'mmkcsim')

    # Se il file non esiste, compila con make
    if not os.path.isfile(SIMULATOR):
        print(f"[INFO] Compilatore '{SIMULATOR}' non trovato. Eseguo 'make' in {SIMULATOR_DIR}...")
        try:
            subprocess.run(['make'], cwd=SIMULATOR_DIR, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Compilazione fallita in {SIMULATOR_DIR}: {e}")

    if parallelism is not None and WARNING is False: 
        print(f"[WARNING] Parallelismo impostato a {parallelism}")
        WARNING = True
    elif parallelism is None and WARNING is False:
        parallelism = multiprocessing.cpu_count()
        print(f"[WARNING] Parallelismo non impostato, verranno utilizzati tutti i core della CPU: {parallelism}")
        WARNING = True

    # Scrive i modelli su file temporaneo in JSON
    with tempfile.NamedTemporaryFile(mode='w', delete=True, suffix='.json') as temp_file:
        temp_file.write(ModelEncoder().encode(models))
        temp_file.flush()

        # Comando finale da eseguire
        cmd = f"{SIMULATOR} {temp_file.name} {n_arrivals:d} {seed} {parallelism}"
        print(f"[INFO] Lancio simulazione: {cmd}")
        proc_result = run_process_subprocess_run(cmd)

        # Parse dei risultati
        if proc_result:
            return json.loads(proc_result.stdout)
        else:
            print("[WARN] Nessun risultato, contenuto file temporaneo:")
            with open(temp_file.name, "r") as jsonf:
                print(jsonf.read())
            return None



# Questo metodo serve a creare dei tassi di arrivo che ripsettino il massimo carico del sistema
def create_random_X (rng, model, samples, max_rho=1.1):
    n = len(model.serv_times)
    max_lambda = max_rho*model.memory/np.array(model.serv_times)/np.array(model.mem_demands)/n
    min_lambda = 0.005/np.array(model.serv_times)/n

    X = np.zeros((samples, n))
    for i in range(n):
        X[:,i] = rng.uniform(min_lambda[i], max_lambda[i], samples)
    return X


def add_random_X(rng, model, X, samples = 1000, max_rho=1.1):

    if X.shape[0] > samples: 
        X = X[:samples, :]
    elif X.shape[0] < samples:
        raise ValueError(
            f"X ha meno righe ({X.shape[0]}) rispetto a samples ({samples})."
        )

    _, n_minus_1 = X.shape
    n = n_minus_1 + 1

    # Calcola utilizzo (rho) per ciascuna funzione
    util_mat = np.array(model.mem_demands) * np.array(model.serv_times) / model.memory

    # Calcola carico attuale per ogni sample (considera le funzioni presenti e non quella aggiunta)
    rho_old = np.sum(X * util_mat[:-1], axis=1)

    # Carico residuo per ogni sample
    rho_remaining = np.clip(max_rho - rho_old, 0, None)

    # Calcolo limiti tasso per la nuova funzione
    mem_new = model.mem_demands[-1]
    serv_new = model.serv_times[-1]

    max_lambda_new = max_rho * model.memory / (serv_new * mem_new * n)
    min_lambda_new = 0.005 / (serv_new * n)

    # Aggiusta max lambda per ogni sample in base al carico residuo
    max_lambda_new_adjusted = np.minimum(rho_remaining / util_mat[-1], max_lambda_new)

    # Genera nuovi tassi validi per B per ogni sample
    max_lambda_new_adjusted = np.maximum(max_lambda_new_adjusted, min_lambda_new + 1e-8) # Serve solo per controllo di coerenza, rappresenta un caso limite

    lambda_new = rng.uniform(min_lambda_new, max_lambda_new_adjusted)

    # Ritorna matrice con la nuova colonna
    return np.hstack([X, lambda_new.reshape(-1, 1)])
