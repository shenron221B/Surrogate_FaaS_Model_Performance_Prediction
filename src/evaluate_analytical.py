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
    from models.model import model_from_conf
    from models.kaufman import ExactKaufman
    from models.mmck import MMckModel
    from utils.time_logger import log_execution_time
except ImportError as e:
    print(f"[ERRORE] Impossibile importare i moduli analitici: {e}")
    sys.exit(1)

def print_usage():
    print("Uso: python3 evaluate_analytical.py <dir_risultati>")
    sys.exit(1)

def calcola_errori_array(Y_true, Y_pred, is_utility=False):
    """Calcola l'array degli errori riga per riga per i boxplot"""
    if is_utility:
        # RMSE: media degli errori quadrati sulle funzioni, poi radice
        mse = np.mean((Y_true - Y_pred) ** 2, axis=1)
        return np.sqrt(mse)
    else:
        # MAPE: errore percentuale medio sulle funzioni
        Y_pred = np.maximum(Y_pred, 0.001)
        with np.errstate(divide='ignore', invalid='ignore'):
            perc_err = np.abs((Y_true - Y_pred) / np.where(Y_true == 0, 1, Y_true)) * 100
        return np.mean(perc_err, axis=1)

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
        config_data = yaml.safe_load(f)
    
    base_model = model_from_conf(config_data)
    data_real = np.load(real_dataset_file)
    min_len = min(data_real['X'].shape[0], data_real['RT'].shape[0])
    
    X_real = data_real['X'][:min_len]
    RT_real = data_real['RT'][:min_len]
    U_real = data_real['U'][:min_len]

    mmck_model = MMckModel(base_model)
    kaufman_model = ExactKaufman(base_model)

    start_mmck = time.time()
    RT_mmck, U_mmck = mmck_model.predict(X_real)
    time_mmck = time.time() - start_mmck
    log_execution_time(base_dir, "MMcK", time_mmck)

    start_kauf = time.time()
    RT_kauf, U_kauf = kaufman_model.predict(X_real)
    time_kauf = time.time() - start_kauf
    log_execution_time(base_dir, "Kaufman", time_kauf)

    net_ov = config_data.get('net_overhead', 0.0)
    RT_mmck += net_ov
    RT_kauf += net_ov

    # calcolo array errori
    arr_rt_mmck = calcola_errori_array(RT_real, RT_mmck, is_utility=False)
    arr_u_mmck = calcola_errori_array(U_real, U_mmck, is_utility=True)
    arr_rt_kauf = calcola_errori_array(RT_real, RT_kauf, is_utility=False)
    arr_u_kauf = calcola_errori_array(U_real, U_kauf, is_utility=True)

    # salvataggio CSV originale
    df_data = {}
    for f in range(X_real.shape[1]):
        df_data[f'Lambda_F{f+1}'] = X_real[:, f]
        df_data[f'RT_Real_F{f+1}'] = RT_real[:, f]
        df_data[f'RT_MMcK_F{f+1}'] = RT_mmck[:, f]
        df_data[f'RT_Kaufman_F{f+1}'] = RT_kauf[:, f]
        df_data[f'U_Real_F{f+1}'] = U_real[:, f]
        df_data[f'U_MMcK_F{f+1}'] = U_mmck[:, f]
        df_data[f'U_Kaufman_F{f+1}'] = U_kauf[:, f]

    pd.DataFrame(df_data).to_csv(os.path.join(base_dir, "analytical_results.csv"), index=False)

    # salvataggio JSON per i Boxplot
    json_data = {
        "RT": {
            "M/M/c/K": arr_rt_mmck.tolist(),
            "Kaufman": arr_rt_kauf.tolist()
        },
        "U": {
            "M/M/c/K": arr_u_mmck.tolist(),
            "Kaufman": arr_u_kauf.tolist()
        }
    }
    json_out = os.path.join(base_dir, "analytical_errors.json")
    with open(json_out, "w") as f:
        json.dump(json_data, f)
    
    print(f"[OK] Salvato array errori in: {json_out}")

if __name__ == "__main__":
    main()
