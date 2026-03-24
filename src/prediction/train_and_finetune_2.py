import sys
import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import subprocess

# =========================================================================
# CONFIGURAZIONE DINAMICA
# =========================================================================
# chiavi disponibili nel .npz:
# 'X', 'Warm', 'Cold', 'Success', 'Queue', 'Init', 'NetOv',
# 'PoolMem', 'FuncMem', 'Cpus', 'FuncCpu', 'QueueLen', 'RT', 'U'
FEATURES_TO_USE = ['X']
TARGETS_TO_PREDICT = ['RT', 'U']

USE_STRATIFICATION = True
FT_SAMPLES = [5, 15, 25, 50]

# Parametri Architettura NN
HIDDEN_UNITS = [120, 80]
EPOCHS_FT = 50  # Ridotto per evitare Overfitting catastrofico su pochi dati
L2_REG = 0.0001  # REINSERITO: Previene l'esplosione dei pesi (Regolarizzazione)


# =========================================================================

def print_usage():
    print("Uso: python3 train_and_finetune_2.py <path_reale.npz> <path_sintetico.npz>")
    sys.exit(1)


def load_dynamic_data(path, features, targets):
    data = np.load(path)
    X_list = []
    for f in features:
        mat = data[f]
        if len(mat.shape) == 1: mat = mat.reshape(-1, 1)
        X_list.append(mat)
    X_combined = np.hstack(X_list)

    Y_dict = {}
    for t in targets:
        mat = data[t]
        if len(mat.shape) == 1: mat = mat.reshape(-1, 1)
        Y_dict[t] = mat

    return X_combined, Y_dict


def calculate_dynamic_errors(Y_true_dict, Y_pred_dict):
    errors_summary = {}
    for target in TARGETS_TO_PREDICT:
        y_true = Y_true_dict[target]
        y_pred = Y_pred_dict[target]

        if target in ['U', 'Success', 'Warm', 'Cold']:
            y_pred = np.clip(y_pred, 0.0, 1.0)
            rmse_array = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
            errors_summary[target] = {"mean": np.mean(rmse_array), "arr": rmse_array.tolist(), "type": "RMSE"}
        else:
            y_pred = np.maximum(y_pred, 0.001)
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_err = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)) * 100
            mape_array = np.mean(perc_err, axis=1)
            errors_summary[target] = {"mean": np.mean(mape_array), "arr": mape_array.tolist(), "type": "MAPE"}
    return errors_summary


def find_closest_synthetic_dynamic(X_test_scaled, X_syn_scaled, Y_syn_dict):
    Y_sim_matched = {t: [] for t in TARGETS_TO_PREDICT}
    for x_test in X_test_scaled:
        distances = np.linalg.norm(X_syn_scaled - x_test, axis=1)
        closest_idx = np.argmin(distances)
        for t in TARGETS_TO_PREDICT:
            Y_sim_matched[t].append(Y_syn_dict[t][closest_idx])

    for t in TARGETS_TO_PREDICT:
        Y_sim_matched[t] = np.array(Y_sim_matched[t])
    return Y_sim_matched


def build_dynamic_nn(input_shape, targets_dict):
    inputs = layers.Input(shape=(input_shape,))
    # Aggiunta Regolarizzazione L2!
    x = layers.Dense(HIDDEN_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REG))(inputs)
    x = layers.Dense(HIDDEN_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REG))(x)

    outputs = []
    for target_name, target_mat in targets_dict.items():
        num_funcs = target_mat.shape[1]
        if target_name in ['U', 'Success', 'Warm', 'Cold']:
            out = layers.Dense(num_funcs, activation='sigmoid', name=target_name)(x)
        else:
            out = layers.Dense(num_funcs, activation='softplus', name=target_name)(x)
        outputs.append(out)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


def dict_to_list(y_dict):
    return [y_dict[t] for t in TARGETS_TO_PREDICT]


def list_to_dict(y_list):
    if len(TARGETS_TO_PREDICT) == 1:
        y_list = [y_list]
    return {TARGETS_TO_PREDICT[i]: y_list[i] for i in range(len(TARGETS_TO_PREDICT))}


def main():
    if len(sys.argv) < 3:
        print_usage()

    real_path = os.path.abspath(sys.argv[1])
    syn_path = os.path.abspath(sys.argv[2])

    real_name = os.path.splitext(os.path.basename(real_path))[0]
    syn_name = os.path.splitext(os.path.basename(syn_path))[0]
    base_dir = os.path.dirname(real_path)

    ml_dir = os.path.join(base_dir, "ML_Dynamic")
    os.makedirs(ml_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f" PIPELINE ML DINAMICA (Con Regolarizzazione e Scaling)")
    print(f" INPUT FEATURES : {FEATURES_TO_USE}")
    print(f" TARGET OUTPUTS : {TARGETS_TO_PREDICT}")
    print(f"{'=' * 70}\n")

    X_syn, Y_syn = load_dynamic_data(syn_path, FEATURES_TO_USE, TARGETS_TO_PREDICT)
    X_real, Y_real = load_dynamic_data(real_path, FEATURES_TO_USE, TARGETS_TO_PREDICT)

    # --- SCALING DINAMICO DEI TARGET (Per evitare esplosione gradienti) ---
    target_scalers = {}
    for t in TARGETS_TO_PREDICT:
        if t not in ['U', 'Success', 'Warm', 'Cold']:
            max_val = np.max(Y_syn[t]) if np.max(Y_syn[t]) > 0 else 1.0
            target_scalers[t] = max_val
            Y_syn[t] = Y_syn[t] / max_val
            Y_real[t] = Y_real[t] / max_val

    stratify_labels = None
    if USE_STRATIFICATION:
        real_data_full = np.load(real_path)
        if 'U' in real_data_full.files:
            U_matrix = real_data_full['U']
            stratify_labels = np.all(U_matrix > 0.90, axis=1).astype(int)
            sani = np.sum(stratify_labels == 1)
            degradati = np.sum(stratify_labels == 0)
            print(f" -> [STRATIFICAZIONE] Trovati {sani} scenari 'Sani' (U > 90%) e {degradati} 'Degradati'")
            if sani < 2 or degradati < 2:
                print(" -> [WARN] Classi troppo sbilanciate. Stratificazione disabilitata.")
                stratify_labels = None
        else:
            stratify_labels = None

    scaler = StandardScaler()
    split_indices = train_test_split(np.arange(len(X_real)), test_size=0.4, random_state=42, stratify=stratify_labels)
    train_idx, test_idx = split_indices[0], split_indices[1]

    X_pool = X_real[train_idx]
    X_test = X_real[test_idx]

    Y_pool = {t: Y_real[t][train_idx] for t in TARGETS_TO_PREDICT}
    # Per il Test, teniamo i valori ORIGINALI NON SCALATI per calcolare l'errore reale
    Y_test_unscaled = load_dynamic_data(real_path, FEATURES_TO_USE, TARGETS_TO_PREDICT)[1]
    Y_test = {t: Y_test_unscaled[t][test_idx] for t in TARGETS_TO_PREDICT}

    X_syn_scaled = scaler.fit_transform(X_syn)
    X_pool_scaled = scaler.transform(X_pool)
    X_test_scaled = scaler.transform(X_test)

    boxplot_data = {t: {"type": "", "models": {}} for t in TARGETS_TO_PREDICT}
    report_lines = [f"REPORT ERRORI DINAMICO: {real_name} vs {syn_name}\n"]

    # --- 1. SIMULATORE PURO ---
    print("--- 1. Valutazione Simulatore Puro ---")
    # Leggiamo i valori NON scalati dal file sintetico per il confronto diretto
    _, Y_syn_unscaled = load_dynamic_data(syn_path, FEATURES_TO_USE, TARGETS_TO_PREDICT)
    Y_sim_matched = find_closest_synthetic_dynamic(X_test_scaled, X_syn_scaled, Y_syn_unscaled)
    sim_errors = calculate_dynamic_errors(Y_test, Y_sim_matched)

    report_lines.append("Simulatore Puro:")
    for t in TARGETS_TO_PREDICT:
        boxplot_data[t]["type"] = sim_errors[t]["type"]
        boxplot_data[t]["models"]["Simulatore"] = sim_errors[t]["arr"]
        err_str = f"  -> {t} {sim_errors[t]['type']}: {sim_errors[t]['mean']:.2f}{'%' if sim_errors[t]['type'] == 'MAPE' else ''}"
        report_lines.append(err_str)
        print(err_str)

    # --- 2. TRAINING BASE SINTETICO ---
    print("\n--- 2. Addestramento moNN Base su Dati Sintetici (Zero-Shot) ---")
    base_nn = build_dynamic_nn(X_syn_scaled.shape[1], Y_syn)
    base_nn.fit(X_syn_scaled, dict_to_list(Y_syn), epochs=200, batch_size=32, verbose=0)

    preds_base_scaled = list_to_dict(base_nn.predict(X_test_scaled, verbose=0))
    # Riportiamo le predizioni alla scala reale prima di calcolare l'errore
    for t in target_scalers: preds_base_scaled[t] *= target_scalers[t]

    base_errors = calculate_dynamic_errors(Y_test, preds_base_scaled)

    report_lines.append("\nML Sintetico (Zero-Shot):")
    for t in TARGETS_TO_PREDICT:
        boxplot_data[t]["models"]["ML_Sintetico"] = base_errors[t]["arr"]
        err_str = f"  -> {t} {base_errors[t]['type']}: {base_errors[t]['mean']:.2f}{'%' if base_errors[t]['type'] == 'MAPE' else ''}"
        report_lines.append(err_str)
        print(err_str)

    # --- 3. FINE-TUNING INCREMENTALE ---
    print("\n--- 3. Fine-Tuning Incrementale sul Dataset Reale ---")
    valid_ft_samples = sorted([s for s in FT_SAMPLES if s <= len(X_pool)])

    for n in valid_ft_samples:
        X_ft = X_pool_scaled[:n]
        Y_ft_list = dict_to_list({t: Y_pool[t][:n] for t in TARGETS_TO_PREDICT})

        ft_model = tf.keras.models.clone_model(base_nn)
        ft_model.set_weights(base_nn.get_weights())
        ft_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')

        batch_s = max(1, n // 2)
        ft_model.fit(X_ft, Y_ft_list, epochs=EPOCHS_FT, batch_size=batch_s, verbose=0)

        preds_ft_scaled = list_to_dict(ft_model.predict(X_test_scaled, verbose=0))
        for t in target_scalers: preds_ft_scaled[t] *= target_scalers[t]

        ft_errors = calculate_dynamic_errors(Y_test, preds_ft_scaled)

        label = f"FT_{n}pt"
        report_lines.append(f"\nFine-Tuning ({label}):")
        print(f"  -> {label}:", end=" ")
        for t in TARGETS_TO_PREDICT:
            boxplot_data[t]["models"][label] = ft_errors[t]["arr"]
            err_str = f"| {t}: {ft_errors[t]['mean']:.2f}{'%' if ft_errors[t]['type'] == 'MAPE' else ''} "
            report_lines.append(err_str)
            print(err_str, end="")
        print()

    txt_report_name = f"ML_Report_{real_name}_{syn_name}.txt"
    with open(os.path.join(ml_dir, txt_report_name), "w") as f:
        f.write("\n".join(report_lines))

    json_path = os.path.join(ml_dir, "boxplot_data.json")
    with open(json_path, "w") as f:
        json.dump(boxplot_data, f)

    print("\n[OK] Modelli e Report salvati! Generazione grafici in corso...")

    # Path Dinamico: Cerca plot_errors.py nella stessa cartella di questo script
    plot_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_errors.py")
    if os.path.exists(plot_script):
        subprocess.run(["python3", plot_script, json_path])
    else:
        print(f"[WARN] Script plot_errors.py non trovato in {os.path.dirname(os.path.abspath(__file__))}.")


if __name__ == "__main__":
    main()