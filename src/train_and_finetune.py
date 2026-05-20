import sys
import os
import numpy as np
import json
import yaml
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import subprocess
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'utils'))
try:
    # Import della funzione custom per loggare i tempi di esecuzione
    from time_logger import log_execution_time
except ImportError:
    print("[WARN] Impossibile importare time_logger. Assicurati che esista in src/utils/")


    def log_execution_time(*args, **kwargs):
        pass

tf.get_logger().setLevel('ERROR')

# feature di input (X = tassi di arrivo delle funzioni, es. lambda_1, lambda_2)
FEATURES_TO_USE = ['X']
# target che la rete neurale dovrà prevedere (variabili di output)
TARGETS_TO_PREDICT = ['RT', 'U', 'Cold', 'Success']
# se True, tenta di bilanciare il dataset di test rispetto a una metrica
USE_STRATIFICATION = False
# numero di campioni reali usati durante la fase di Fine-Tuning
FT_SAMPLES = [5, 15, 20, 25]
# architettura della Rete Neurale: due layer nascosti
HIDDEN_UNITS = [32, 16]
# numero di epoche di addestramento per la fase di Fine-Tuning
EPOCHS_FT = 400
# parametro di regolarizzazione L2 (Ridge) per stabilizzare i pesi durante il FT con pochi dati
L2_REG = 0.00001


def print_usage():
    print("Uso: python3 train_and_finetune.py <path_reale.npz> <path_sintetico.npz>")
    sys.exit(1)


def XtoRho_from_dict(config_dict, X_col):
    """
    Converte il tasso di arrivo (lambda)
    in un fattore di utilizzazione della memoria (rho).
    Formula: rho = lambda * tempo_servizio * memoria_richiesta / memoria_totale
    Questo aiuta la Rete Neurale a 'capire' quanto il nodo sia vicino alla saturazione.
    """
    serv_times = config_dict.get('serv_time_duration', [])
    mem_demands = config_dict.get('mem_demands', [])
    system_memory = config_dict.get('system_memory', 2048)

    N = len(serv_times)
    # calcolo del costo unitario in memoria per ogni funzione
    util_mat = np.array(mem_demands) * np.array(serv_times) / system_memory

    X_rho = np.zeros_like(X_col)
    cols_to_transform = min(N, X_col.shape[1])
    # calcolo della densità di carico (rho)
    X_rho[:, :cols_to_transform] = X_col[:, :cols_to_transform] * util_mat[:cols_to_transform]
    return X_rho


def calculate_dynamic_errors(Y_true_dict, Y_pred_dict):
    """
    Calcola l'errore predittivo basato sulla natura della metrica.
    - Metriche probabilistiche (U, Success, Cold): Root Mean Squared Error (RMSE)
    - Metriche temporali (RT): Mean Absolute Percentage Error (MAPE)
    """
    errors_summary = {}
    for target in TARGETS_TO_PREDICT:
        y_true = Y_true_dict[target]
        y_pred = Y_pred_dict[target]

        # gestione probabilità (vincolate tra 0% e 100%)
        if target in ['U', 'Success', 'Warm', 'Cold']:
            y_pred = np.clip(y_pred, 0.0, 1.0)  # evita che la rete dia probabilità <0 o >1
            rmse_array = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
            errors_summary[target] = {"mean": np.mean(rmse_array), "arr": rmse_array.tolist(), "type": "RMSE"}

        # gestione tempi
        else:
            y_pred = np.maximum(y_pred, 0.001)  # evita divisioni per zero
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_err = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)) * 100
            mape_array = np.mean(perc_err, axis=1)
            errors_summary[target] = {"mean": np.mean(mape_array), "arr": mape_array.tolist(), "type": "MAPE"}
    return errors_summary


def find_closest_synthetic_dynamic(X_test_scaled, X_syn_scaled, Y_syn_dict):
    """
    Simula l'oracolo del 'Simulatore Puro'.
    Per ogni punto di test reale, cerca nella matrice di addestramento sintetica
    il punto con i carichi (X) più simili (distanza euclidea minima).
    Restituisce i valori simulati (Y) di quel punto per confrontarli con la realtà.
    """
    Y_sim_matched = {t: [] for t in TARGETS_TO_PREDICT}
    for x_test in X_test_scaled:
        # calcola la distanza tra il punto di test e tutti i punti simulati
        distances = np.linalg.norm(X_syn_scaled - x_test, axis=1)
        closest_idx = np.argmin(distances)  # prende l'indice del punto più vicino

        for t in TARGETS_TO_PREDICT:
            Y_sim_matched[t].append(Y_syn_dict[t][closest_idx])

    for t in TARGETS_TO_PREDICT:
        Y_sim_matched[t] = np.array(Y_sim_matched[t])
    return Y_sim_matched


# ===============================
# ARCHITETTURA DELLA RETE NEURALE
# ===============================

def build_single_target_nn(input_shape, target_dim, target_name):
    """
    Costruisce una rete Multi-Layer Perceptron (MLP) dedicata a una specifica metrica.
    Si usano reti separate per evitare interferenze nell'apprendimento di metriche
    che hanno scale fisiche diverse (es. secondi vs probabilità).
    """
    inputs = layers.Input(shape=(input_shape,))

    # layer nascosti con attivazione ReLU e regolarizzazione L2 per la stabilità
    x = layers.Dense(HIDDEN_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REG), name="hidden_1")(inputs)
    x = layers.Dense(HIDDEN_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REG), name="hidden_2")(x)

    # livello di uscita e funzione di Loss adattati al tipo di metrica
    if target_name in ['RT', 'Queue', 'Init', 'NetOv']:
        # 'softplus' garantisce previsioni temporali fluide e sempre positive (>0)
        out = layers.Dense(target_dim, activation='softplus', name=f"out_{target_name}")(x)
        loss_fn = 'mean_absolute_percentage_error'
    else:
        # 'sigmoid' garantisce che le previsioni probabilistiche siano comprese tra [0, 1]
        out = layers.Dense(target_dim, activation='sigmoid', name=f"out_{target_name}")(x)
        loss_fn = 'mse'

    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=loss_fn)
    return model


# ==============================
# FLUSSO PRINCIPALE D'ESECUZIONE
# ==============================

def main():
    if len(sys.argv) < 3:
        print_usage()

    # parsing dei percorsi dei file in input (Dati Reali e Dati Sintetici)
    real_path = os.path.abspath(sys.argv[1])
    syn_path = os.path.abspath(sys.argv[2])
    real_name = os.path.splitext(os.path.basename(real_path))[0]
    syn_name = os.path.splitext(os.path.basename(syn_path))[0]
    base_dir = os.path.dirname(real_path)

    # creazione della cartella per salvare i report e i modelli ML
    ml_dir = os.path.join(base_dir, "ML_Test_Fix")
    os.makedirs(ml_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f" PIPELINE ML: (Reti Separate + MAPE Loss + Polinomi + FREEZING PARZIALE)")
    print(f" FEATURES: {FEATURES_TO_USE}")
    print(f"{'=' * 70}\n")

    # caricamento delle matrici .npz
    data_syn = np.load(syn_path)
    data_real_raw = np.load(real_path)

    # allineamento dimensioni
    min_len = min([data_real_raw[k].shape[0] for k in data_real_raw.files])
    print(f"[INFO] Dimensioni analizzate. Tronco il dataset reale a {min_len} campioni utili.")
    data_real = {k: data_real_raw[k][:min_len] for k in data_real_raw.files}

    # lettura della configurazione del simulatore
    config_dict = None
    yml_path = os.path.join(base_dir, "simulator-conf.yml")
    if os.path.exists(yml_path):
        with open(yml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

    # preparazione matrici input (X)
    X_syn_parts = []
    X_real_parts = []

    for f in FEATURES_TO_USE:
        s_mat, r_mat = data_syn[f], data_real[f]
        if len(s_mat.shape) == 1: s_mat = s_mat.reshape(-1, 1)
        if len(r_mat.shape) == 1: r_mat = r_mat.reshape(-1, 1)

        # applicazione della conversione X -> Rho
        if f == 'X' and config_dict:
            s_mat = XtoRho_from_dict(config_dict, s_mat)
            r_mat = XtoRho_from_dict(config_dict, r_mat)

        X_syn_parts.append(s_mat)
        X_real_parts.append(r_mat)

    X_syn = np.hstack(X_syn_parts)
    X_real = np.hstack(X_real_parts)

    # preparazione matrici output (Y)
    Y_syn = {t: data_syn[t] if len(data_syn[t].shape) > 1 else data_syn[t].reshape(-1, 1) for t in TARGETS_TO_PREDICT}
    Y_real = {t: data_real[t] if len(data_real[t].shape) > 1 else data_real[t].reshape(-1, 1) for t in
              TARGETS_TO_PREDICT}

    # split Train/Test sul dataset reale
    stratify_labels = None
    if USE_STRATIFICATION and 'U' in data_real.keys():
        stratify_labels = np.all(data_real['U'] > 0.50, axis=1).astype(int)

    split_indices = train_test_split(np.arange(len(X_real)), test_size=0.4, random_state=42, stratify=stratify_labels)
    train_idx, test_idx = split_indices[0], split_indices[1]

    # X_pool rappresenta la libreria di dati reali da cui pescare i 5-25 punti per il Fine-Tuning
    X_pool = X_real[train_idx]
    X_test = X_real[test_idx]
    Y_pool = {t: Y_real[t][train_idx] for t in TARGETS_TO_PREDICT}
    Y_test = {t: Y_real[t][test_idx] for t in TARGETS_TO_PREDICT}

    # ===============================================================
    # PRE-PROCESSING (Standardizzazione + Trasformazione Polinomiale)
    # ===============================================================
    # 1. Standardizzazione (media=0, std=1) per stabilità numerica
    scaler = StandardScaler()
    # 2. Polinomi di Grado 3: Genera combinazioni non-lineari dei carichi
    poly = PolynomialFeatures(degree=3, include_bias=False)

    X_syn_poly = poly.fit_transform(scaler.fit_transform(X_syn))
    X_pool_poly = poly.transform(scaler.transform(X_pool))
    X_test_poly = poly.transform(scaler.transform(X_test))

    # struttura per immagazzinare i risultati per i boxplot
    boxplot_data = {t: {"type": "", "models": {}} for t in TARGETS_TO_PREDICT}
    report_lines = [f"REPORT ERRORI FIX: {real_name} vs {syn_name}\n"]

    # --------------------------------------------------
    # 1. Valutazione Errore Baseline del Simulatore Puro
    # --------------------------------------------------
    print("--- 1. Valutazione Simulatore Puro ---")
    X_syn_scaled = scaler.transform(X_syn)
    X_test_scaled = scaler.transform(X_test)
    # cerca il punto sintetico più simile e ne preleva l'errore
    Y_sim_matched = find_closest_synthetic_dynamic(X_test_scaled, X_syn_scaled, Y_syn)
    sim_errors = calculate_dynamic_errors(Y_test, Y_sim_matched)

    report_lines.append("Simulatore Puro:")
    for t in TARGETS_TO_PREDICT:
        boxplot_data[t]["type"] = sim_errors[t]["type"]
        boxplot_data[t]["models"]["Simulatore"] = sim_errors[t]["arr"]
        err_str = f"  -> {t} {sim_errors[t]['type']}: {sim_errors[t]['mean']:.2f}{'%' if sim_errors[t]['type'] == 'MAPE' else ''}"
        report_lines.append(err_str)
        print(err_str)

    # inizio conteggio dei tempi computazionali
    training_start_time = time.time()

    # ---------------------------------------------------------
    # 2. Addestramento BASE (ZERO-SHOT) sui Dati del Simulatore
    # ---------------------------------------------------------
    print("\n--- 2. Addestramento ZERO-SHOT ---")
    base_models = {}
    preds_base = {}

    for t in TARGETS_TO_PREDICT:
        print(f"  -> Addestramento modello per {t}")
        # creazione e fit su dati sintetici
        model_t = build_single_target_nn(X_syn_poly.shape[1], Y_syn[t].shape[1], t)
        model_t.fit(X_syn_poly, Y_syn[t], epochs=200, batch_size=32, verbose=0)

        base_models[t] = model_t
        preds_base[t] = model_t.predict(X_test_poly, verbose=0)

    base_errors = calculate_dynamic_errors(Y_test, preds_base)

    report_lines.append("\nML Sintetico (Zero-Shot):")
    for t in TARGETS_TO_PREDICT:
        boxplot_data[t]["models"]["ML_Sintetico"] = base_errors[t]["arr"]
        err_str = f"  -> {t} {base_errors[t]['type']}: {base_errors[t]['mean']:.2f}{'%' if base_errors[t]['type'] == 'MAPE' else ''}"
        report_lines.append(err_str)
        print(err_str)

    # ------------------------------------------
    # 3. Fine-Tuning Incrementale sui Dati Reali
    # ------------------------------------------
    print("\n--- 3. Fine-Tuning Incrementale ---")
    valid_ft_samples = sorted([s for s in FT_SAMPLES if s <= len(X_pool)])

    for n in valid_ft_samples:
        # preleva solo n campioni dal pool di dati reali
        X_ft = X_pool_poly[:n]
        preds_ft = {}

        for t in TARGETS_TO_PREDICT:
            # clona il modello Zero-Shot e ne recupera i pesi appresi sul simulatore
            ft_model_t = tf.keras.models.clone_model(base_models[t])
            ft_model_t.set_weights(base_models[t].get_weights())

            # FREEZING PARZIALE
            for layer in ft_model_t.layers:
                if layer.name == "hidden_1":
                    layer.trainable = False

            loss_fn = 'mean_absolute_percentage_error' if t in ['RT', 'Queue', 'Init', 'NetOv'] else 'mse'
            ft_model_t.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=loss_fn)

            # addestramento Fine-Tuning
            batch_s = max(1, n // 2)
            ft_model_t.fit(X_ft, Y_pool[t][:n], epochs=EPOCHS_FT, batch_size=batch_s, verbose=0)

            # predizione sul dataset di test
            preds_ft[t] = ft_model_t.predict(X_test_poly, verbose=0)

        ft_errors = calculate_dynamic_errors(Y_test, preds_ft)

        # salvataggio metriche
        label = f"FT_{n}pt"
        report_lines.append(f"\nFine-Tuning ({label}):")
        print(f"  -> {label}:", end=" ")
        for t in TARGETS_TO_PREDICT:
            boxplot_data[t]["models"][label] = ft_errors[t]["arr"]
            err_str = f"| {t}: {ft_errors[t]['mean']:.2f}{'%' if ft_errors[t]['type'] == 'MAPE' else ''} "
            report_lines.append(err_str)
            print(err_str, end="")
        print()

    # ===================================================
    # 5. ESPORTAZIONE DATI E CALCOLO COSTI COMPUTAZIONALI
    # ===================================================

    txt_report_name = f"ML_Report_{real_name}_{syn_name}.txt"
    with open(os.path.join(ml_dir, txt_report_name), "w") as f:
        f.write("\n".join(report_lines))

    json_path = os.path.join(ml_dir, "boxplot_data.json")
    with open(json_path, "w") as f:
        json.dump(boxplot_data, f)

    print("\n[OK] Modelli e Report salvati! Generazione grafici in corso...")

    training_time_seconds = time.time() - training_start_time

    sim_time_seconds = 0
    times_json_path = os.path.join(base_dir, "execution_times.json")
    if os.path.exists(times_json_path):
        try:
            with open(times_json_path, 'r') as f:
                times_data = json.load(f)
                sim_time_seconds = times_data.get("Simulatore", {}).get("total_time_seconds", 0)
        except Exception as e:
            print(f"[WARN] Impossibile leggere il tempo del simulatore: {e}")

    TIME_PER_ROW_SECONDS = 660

    max_ft_points = max(valid_ft_samples) if valid_ft_samples else 0
    data_collection_time = TIME_PER_ROW_SECONDS * max_ft_points

    total_ml_cost = sim_time_seconds + data_collection_time + training_time_seconds

    breakdown = {
        "simulation_time_sec": sim_time_seconds,
        "data_collection_time_sec": data_collection_time,
        "training_time_sec": training_time_seconds
    }

    log_execution_time(base_dir, "ML_FineTuning", total_ml_cost, breakdown)

    plot_script = "/root/tesi_project/src/plot/plot_errors.py"

    if os.path.exists(plot_script):
        subprocess.run(["python3", plot_script, json_path])
    else:
        print(f"[ERRORE] Non ho trovato lo script di plotting in: {plot_script}")


if __name__ == "__main__":
    main()