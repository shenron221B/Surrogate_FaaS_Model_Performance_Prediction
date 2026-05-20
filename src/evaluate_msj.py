import sys
import os
import yaml
import json
import numpy as np
import pandas as pd
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

try:
    from models.grosov23perf import compute_approx_rt
    from utils.time_logger import log_execution_time
except ImportError as e:
    print(f"[ERRORE] Impossibile importare grosov23perf.py: {e}")
    sys.exit(1)

def print_usage():
    print("Uso: python3 evaluate_msj.py <dir_risultati>")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    conf_file = os.path.join(base_dir, "simulator-conf.yml")
    real_dataset_file = os.path.join(base_dir, "dataset.npz")

    if not os.path.exists(conf_file) or not os.path.exists(real_dataset_file):
        print(f"[ERRORE] File mancanti in {base_dir}")
        sys.exit(1)

    with open(conf_file, 'r') as f:
        conf = yaml.safe_load(f)
    
    mem_demands = conf['mem_demands']
    serv_times = conf['serv_time_duration']
    total_mem = conf['system_memory']
    net_ov = conf.get('net_overhead', 0.0)

    data_real = np.load(real_dataset_file)
    min_len = min(data_real['X'].shape[0], data_real['RT'].shape[0])
    X = data_real['X'][:min_len]
    RT_real_matrix = data_real['RT'][:min_len]

    results = []
    start_msj = time.time()
    for i in range(len(X)):
        lambdas = X[i, :].tolist()
        lambda_tot = sum(lambdas)
        if lambda_tot == 0: continue

        weights = X[i, :] / lambda_tot
        rt_real_global = np.sum(weights * RT_real_matrix[i, :])

        try:
            rt_msj = compute_approx_rt(lambdas, serv_times, mem_demands, total_mem)
            rt_msj += net_ov
            status = "OK"
        except ValueError:
            rt_msj = np.nan
            status = "UNSTABLE"

        error = np.nan
        if status == "OK" and rt_real_global > 0:
            error = np.abs((rt_real_global - rt_msj) / rt_real_global) * 100

        results.append({"lambda_tot": lambda_tot, "rt_real_global": rt_real_global, "rt_msj": rt_msj, "mape": error, "status": status})

    total_msj_time = time.time() - start_msj
    log_execution_time(base_dir, "Grosof", total_msj_time)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, "msj_results_comparison.csv"), index=False)

    valid_df = df.dropna(subset=['mape'])
    
    json_data = {
        "RT": {
            "Grosof (MSJ)": valid_df['mape'].tolist()
        }
    }
    json_out = os.path.join(base_dir, "msj_errors.json")
    with open(json_out, "w") as f:
        json.dump(json_data, f)
        
    print(f"[OK] Salvato array errori (stabili) in: {json_out}")

if __name__ == "__main__":
    main()
