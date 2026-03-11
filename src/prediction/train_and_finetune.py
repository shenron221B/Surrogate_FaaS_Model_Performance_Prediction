import sys
import os
import numpy as np
import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import yaml
import subprocess

sys.path.append(os.path.join(os.getcwd(), 'src'))
from fit_eval.tf_fit import fit_multiout_nn
from fit_eval.utils import XtoRho
from models.model import model_from_conf

# parametri architettura
RT_SCALE = 10.0  # fattore di scaling per i gradienti della loss MSE
HIDDEN_UNITS = [120, 80]
L2_REG = 0.0001
EPOCHS_FT = 200


def print_usage():
    print("Uso: python3 train_and_finetune.py <path_reale.npz> <path_sintetico.npz>")
    print("Esempio: python3 train_and_finetune.py results/1f_2GBpm/q_len_5/del/del/dataset.npz results/1f_2GBpm/q_len_5/del/del/dataset_120_10000.npz")
    sys.exit(1)


def load_data(path):
    data = np.load(path)
    return data['X'], data['RT'], data['U']


def calculate_errors_arrays(rt_true, rt_pred, u_true, u_pred):
    # evita divisione per zero in RT e limitazione U
    rt_pred = np.maximum(rt_pred, 0.001)
    u_pred = np.clip(u_pred, 0.0, 1.0)

    # MAPE (%) per Response Time
    with np.errstate(divide='ignore', invalid='ignore'):
        rt_perc_err = np.abs((rt_true - rt_pred) / np.where(rt_true == 0, 1, rt_true)) * 100
    mape_array = np.mean(rt_perc_err, axis=1)
    mape_mean = np.mean(mape_array)

    # RMSE per Utility
    u_squared_err = (u_true - u_pred) ** 2
    rmse_array = np.sqrt(np.mean(u_squared_err, axis=1))
    rmse_mean = np.mean(rmse_array)

    return mape_mean, mape_array.tolist(), rmse_mean, rmse_array.tolist()


def find_closest_synthetic(X_test_real, X_syn, RT_syn, U_syn):
    RT_sim_matched = []
    U_sim_matched = []

    print("\n[VERIFICA SIMULATORE] Ricerca dei punti sintetici più vicini ai reali di Test:")
    for i, x_real in enumerate(X_test_real):
        distances = np.linalg.norm(X_syn - x_real, axis=1)
        closest_idx = np.argmin(distances)

        x_closest = X_syn[closest_idx]
        distanza_euclidea = distances[closest_idx]

        print(
            f"  Test {i + 1:02d} | Carico Reale: {np.round(x_real, 2)} -> Match Sintetico: {np.round(x_closest, 2)} (Dist: {distanza_euclidea:.3f})")

        RT_sim_matched.append(RT_syn[closest_idx])
        U_sim_matched.append(U_syn[closest_idx])

    return np.array(RT_sim_matched), np.array(U_sim_matched)


def main():
    if len(sys.argv) < 3:
        print_usage()

    real_path = os.path.abspath(sys.argv[1])
    syn_path = os.path.abspath(sys.argv[2])

    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print("Errore: Uno dei file npz non esiste.")
        sys.exit(1)

    real_name = os.path.splitext(os.path.basename(real_path))[0]
    syn_name = os.path.splitext(os.path.basename(syn_path))[0]
    base_dir = os.path.dirname(real_path)

    ml_dir = os.path.join(base_dir, "ML")
    os.makedirs(ml_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f" PIPELINE ML & FINE-TUNING (moNN Classica)")
    print(f"=======================================================\n")

    # caricamento modello base per calcolare rho
    yml_path = os.path.join(base_dir, "simulator-conf.yml")
    with open(yml_path, 'r') as f:
        model_obj = model_from_conf(yaml.safe_load(f))

    X_syn, RT_syn, U_syn = load_data(syn_path)
    X_real, RT_real, U_real = load_data(real_path)

    # stratificazione basata sulla saturazione
    stratify_labels = (X_real[:, 0] > 25.0).astype(int)

    X_pool, X_test, RT_pool, RT_test, U_pool, U_test, _, _ = train_test_split(
        X_real, RT_real, U_real, stratify_labels,
        test_size=0.4,
        random_state=42,
        stratify=stratify_labels
    )

    boxplot_data = {"RT_MAPE": {}, "U_RMSE": {}}
    report_lines = [f"REPORT ERRORI FINALI: {real_name} vs {syn_name}\n"]

    # --- 1. ERRORE SIMULATORE PURO ---
    print("--- 1. Valutazione Simulatore Puro ---")
    RT_sim, U_sim = find_closest_synthetic(X_test, X_syn, RT_syn, U_syn)
    sim_mape_mean, sim_mape_arr, sim_rmse_mean, sim_rmse_arr = calculate_errors_arrays(RT_test, RT_sim, U_test, U_sim)

    boxplot_data["RT_MAPE"]["Simulatore"] = sim_mape_arr
    boxplot_data["U_RMSE"]["Simulatore"] = sim_rmse_arr
    report_lines.append(f"Simulatore Puro -> RT MAPE: {sim_mape_mean:.2f}%, U RMSE: {sim_rmse_mean:.4f}")
    print(f"\n-> Risultato Simulatore Puro: RT MAPE = {sim_mape_mean:.2f}% | U RMSE = {sim_rmse_mean:.4f}")

    # --- 2. TRAINING BASE SINTETICO ---
    print("\n--- 2. Addestramento moNN Base su Dati Sintetici (Zero-Shot) ---")
    RT_syn_scaled = RT_syn / RT_SCALE

    # addestramento della Multi-Output NN
    base_nn, _ = fit_multiout_nn(model_obj, X_syn, RT_syn_scaled, U_syn, hidden_units=HIDDEN_UNITS, l2reg=L2_REG)

    # predittore base (feature polynomiali grado 3 su rho)
    poly = PolynomialFeatures(3)
    poly.fit(XtoRho(model_obj, X_syn[:1]))
    base_predictor = lambda x: np.hsplit(base_nn.predict(poly.transform(XtoRho(model_obj, x)), verbose=0), 2)

    preds_base = base_predictor(X_test)
    ml_mape_mean, ml_mape_arr, ml_rmse_mean, ml_rmse_arr = calculate_errors_arrays(RT_test, preds_base[0] * RT_SCALE,
                                                                                   U_test, preds_base[1])

    boxplot_data["RT_MAPE"]["ML_Sintetico"] = ml_mape_arr
    boxplot_data["U_RMSE"]["ML_Sintetico"] = ml_rmse_arr
    report_lines.append(f"ML Sintetico (Zero-Shot) -> RT MAPE: {ml_mape_mean:.2f}%, U RMSE: {ml_rmse_mean:.4f}")
    print(f"-> Risultato ML Sintetico (Zero-Shot): RT MAPE = {ml_mape_mean:.2f}% | U RMSE = {ml_rmse_mean:.4f}")

    base_nn.save(os.path.join(ml_dir, f"model_base_{syn_name}.h5"))

    # --- 3. FINE-TUNING INCREMENTALE ---
    print("\n--- 3. Fine-Tuning Incrementale sul Dataset Reale ---")
    ft_steps = [int(len(X_pool) * p) for p in [0.2, 0.4, 0.6, 0.8, 1.0]]
    ft_steps = sorted(list(set([s for s in ft_steps if s > 0])))

    for n in ft_steps:
        X_ft, RT_ft, U_ft = X_pool[:n], RT_pool[:n], U_pool[:n]

        # clonazione della rete per ogni step per non sovrascrivere l'addestramento precedente
        ft_model = tf.keras.models.clone_model(base_nn)
        ft_model.set_weights(base_nn.get_weights())

        # sblocco totale della rete
        ft_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

        Y_ft_full = np.concatenate((RT_ft / RT_SCALE, U_ft), axis=1)
        batch_s = max(1, n // 2)

        ft_model.fit(poly.transform(XtoRho(model_obj, X_ft)), Y_ft_full, epochs=EPOCHS_FT, batch_size=batch_s,
                     verbose=0)

        preds_ft = np.hsplit(ft_model.predict(poly.transform(XtoRho(model_obj, X_test)), verbose=0), 2)

        ft_mape_mean, ft_mape_arr, ft_rmse_mean, ft_rmse_arr = calculate_errors_arrays(RT_test, preds_ft[0] * RT_SCALE,
                                                                                       U_test, preds_ft[1])

        label = f"FT_{n}pt"
        boxplot_data["RT_MAPE"][label] = ft_mape_arr
        boxplot_data["U_RMSE"][label] = ft_rmse_arr

        report_lines.append(f"{label} -> RT MAPE: {ft_mape_mean:.2f}%, U RMSE: {ft_rmse_mean:.4f}")
        print(f"  -> {label} | RT MAPE: {ft_mape_mean:.2f}% | U RMSE: {ft_rmse_mean:.4f}")

        if n == ft_steps[-1]:
            ft_model.save(os.path.join(ml_dir, f"model_FT_max_{syn_name}.h5"))

    # salvataggio risultati
    txt_report_name = f"ML_Report_{real_name}_{syn_name}.txt"
    with open(os.path.join(ml_dir, txt_report_name), "w") as f:
        f.write("\n".join(report_lines))

    json_path = os.path.join(ml_dir, "boxplot_data.json")
    with open(json_path, "w") as f:
        json.dump(boxplot_data, f)

    print("\n[OK] Modelli e Report salvati! Generazione grafici in corso...")

    plot_script = os.path.join(os.getcwd(), "plot_errors.py")
    if os.path.exists(plot_script):
        subprocess.run(["python3", plot_script, json_path])
    else:
        print(f"[WARN] Script plot_errors.py non trovato in {os.getcwd()}.")


if __name__ == "__main__":
    main()