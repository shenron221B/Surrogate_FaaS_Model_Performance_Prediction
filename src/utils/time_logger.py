import os
import json
import time


def log_execution_time(results_dir, model_name, duration_seconds, breakdown=None):
    """
    Salva il tempo di esecuzione in un file JSON centralizzato.

    - results_dir: La cartella base (es. results/2f2GB...)
    - model_name: "Simulatore", "MMcK", "Kaufman", "Grosof", "ML_FineTuning"
    - duration_seconds: Tempo totale in secondi
    - breakdown: (Opzionale) Dizionario con i sotto-tempi (es. training, dataset collection)
    """
    json_path = os.path.join(results_dir, "execution_times.json")

    # carica dati esistenti o crea nuovo dizionario
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    data[model_name] = {
        "total_time_seconds": duration_seconds,
        "breakdown": breakdown or {}
    }

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"[TIME LOGGER] Salvato tempo per {model_name}: {duration_seconds:.2f}s in {json_path}")