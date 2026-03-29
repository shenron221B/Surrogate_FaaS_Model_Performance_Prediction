import sys
import os
import numpy as np
import json
import yaml
import itertools
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

tf.get_logger().setLevel('ERROR')

TARGETS_TO_PREDICT = ['RT', 'U']
USE_STRATIFICATION = True
FT_SAMPLE_TO_REPORT = 50
EPOCHS_FT = 150
HIDDEN_UNITS = [120, 80]
L2_REG = 0.0001


def print_usage():
    print("Uso: python3 feature_grid_search.py <path_reale.npz> <path_sintetico.npz>")
    sys.exit(1)


def XtoRho_from_dict(config_dict, X):
    serv_times = config_dict.get('serv_time_duration', [])
    mem_demands = config_dict.get('mem_demands', [])
    system_memory = config_dict.get('system_memory', 2048)
    N = len(serv_times)
    util_mat = np.array(mem_demands) * np.array(serv_times) / system_memory
    X_rho = np.zeros_like(X)
    cols_to_transform = min(N, X.shape[1])
    X_rho[:, :cols_to_transform] = X[:, :cols_to_transform] * util_mat[:cols_to_transform]
    return X_rho


def build_single_target_nn(input_shape, target_dim, target_name):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(HIDDEN_UNITS[0], activation='relu', kernel_regularizer=l2(L2_REG))(inputs)
    x = layers.Dense(HIDDEN_UNITS[1], activation='relu', kernel_regularizer=l2(L2_REG))(x)
    if target_name in ['RT', 'Queue', 'Init', 'NetOv']:
        out = layers.Dense(target_dim, activation='softplus', name=f"out_{target_name}")(x)
        loss_fn = 'mean_absolute_percentage_error'
    else:
        out = layers.Dense(target_dim, activation='sigmoid', name=f"out_{target_name}")(x)
        loss_fn = 'mse'
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=loss_fn)
    return model


def calculate_dynamic_errors(Y_true_dict, Y_pred_dict):
    errors_summary = {}
    for target in TARGETS_TO_PREDICT:
        y_true = Y_true_dict[target]
        y_pred = Y_pred_dict[target]
        if target in ['U', 'Success', 'Warm', 'Cold']:
            y_pred = np.clip(y_pred, 0.0, 1.0)
            rmse_array = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))
            errors_summary[target] = np.mean(rmse_array)
        else:
            y_pred = np.maximum(y_pred, 0.001)
            with np.errstate(divide='ignore', invalid='ignore'):
                perc_err = np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true)) * 100
            errors_summary[target] = np.mean(perc_err)
    return errors_summary


def run_pipeline_for_features(features_to_use, real_path, syn_path, config_dict):
    """Esegue l'intera pipeline per una specifica combinazione di feature"""
    data_syn = np.load(syn_path)
    data_real = np.load(real_path)

    # XtoRho applicata solo a 'X'
    X_syn_parts = []
    X_real_parts = []

    for f in features_to_use:
        s_mat, r_mat = data_syn[f], data_real[f]
        if len(s_mat.shape) == 1: s_mat = s_mat.reshape(-1, 1)
        if len(r_mat.shape) == 1: r_mat = r_mat.reshape(-1, 1)

        if f == 'X' and config_dict:
            s_mat = XtoRho_from_dict(config_dict, s_mat)
            r_mat = XtoRho_from_dict(config_dict, r_mat)

        X_syn_parts.append(s_mat)
        X_real_parts.append(r_mat)

    X_syn = np.hstack(X_syn_parts)
    X_real = np.hstack(X_real_parts)

    Y_syn = {t: data_syn[t] if len(data_syn[t].shape) > 1 else data_syn[t].reshape(-1, 1) for t in TARGETS_TO_PREDICT}
    Y_real = {t: data_real[t] if len(data_real[t].shape) > 1 else data_real[t].reshape(-1, 1) for t in
              TARGETS_TO_PREDICT}

    stratify_labels = None
    if USE_STRATIFICATION and 'U' in data_real.files:
        stratify_labels = np.all(data_real['U'] > 0.90, axis=1).astype(int)

    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=3, include_bias=False)

    split_indices = train_test_split(np.arange(len(X_real)), test_size=0.4, random_state=42, stratify=stratify_labels)
    train_idx, test_idx = split_indices[0], split_indices[1]

    X_pool = X_real[train_idx]
    X_test = X_real[test_idx]
    Y_pool = {t: Y_real[t][train_idx] for t in TARGETS_TO_PREDICT}
    Y_test = {t: Y_real[t][test_idx] for t in TARGETS_TO_PREDICT}

    X_syn_poly = poly.fit_transform(scaler.fit_transform(X_syn))
    X_pool_poly = poly.transform(scaler.transform(X_pool))
    X_test_poly = poly.transform(scaler.transform(X_test))

    # addestramento base
    base_models = {}
    for t in TARGETS_TO_PREDICT:
        model_t = build_single_target_nn(X_syn_poly.shape[1], Y_syn[t].shape[1], t)
        model_t.fit(X_syn_poly, Y_syn[t], epochs=200, batch_size=32, verbose=0)
        base_models[t] = model_t

    # Fine Tuning a FT_SAMPLE_TO_REPORT
    preds_ft = {}
    X_ft = X_pool_poly[:FT_SAMPLE_TO_REPORT]

    for t in TARGETS_TO_PREDICT:
        ft_model = tf.keras.models.clone_model(base_models[t])
        ft_model.set_weights(base_models[t].get_weights())
        loss_fn = 'mean_absolute_percentage_error' if t in ['RT', 'Queue', 'Init', 'NetOv'] else 'mse'
        ft_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=loss_fn)

        batch_s = max(1, FT_SAMPLE_TO_REPORT // 2)
        ft_model.fit(X_ft, Y_pool[t][:FT_SAMPLE_TO_REPORT], epochs=EPOCHS_FT, batch_size=batch_s, verbose=0)
        preds_ft[t] = ft_model.predict(X_test_poly, verbose=0)

    return calculate_dynamic_errors(Y_test, preds_ft)


def main():
    if len(sys.argv) < 3:
        print_usage()

    real_path = os.path.abspath(sys.argv[1])
    syn_path = os.path.abspath(sys.argv[2])
    base_dir = os.path.dirname(real_path)

    out_dir = os.path.join(base_dir, "ML_Feature_Search")
    os.makedirs(out_dir, exist_ok=True)

    config_dict = None
    yml_path = os.path.join(base_dir, "simulator-conf.yml")
    if os.path.exists(yml_path):
        with open(yml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

    # COMBINAZIONI DA TESTARE
    base_feature = ['X']
    optional_features = ['Cold', 'Success', 'Queue']

    combinations_to_test = []
    # genera combinazioni di lunghezza 0, 1, 2 e 3
    for r in range(len(optional_features) + 1):
        for combo in itertools.combinations(optional_features, r):
            combinations_to_test.append(base_feature + list(combo))

    print(f"\n{'=' * 70}")
    print(f" AVVIO FEATURE GRID SEARCH (Architettura Test 7 - 50pt FT)")
    print(f" Testerò {len(combinations_to_test)} combinazioni diverse...")
    print(f"{'=' * 70}\n")

    results = []

    for i, features in enumerate(combinations_to_test):
        feat_str = " + ".join(features)
        print(f"[{i + 1}/{len(combinations_to_test)}] Addestramento combinazione: {feat_str} ... ", end="", flush=True)

        try:
            errors = run_pipeline_for_features(features, real_path, syn_path, config_dict)
            results.append({"features": feat_str, "RT_MAPE": errors['RT'], "U_RMSE": errors['U']})
            print(f"Fatto! (RT: {errors['RT']:.2f}%, U: {errors['U']:.3f})")
        except Exception as e:
            print(f"ERRORE: {str(e)}")

    print("\n\n" + "=" * 70)
    print("🏆 CLASSIFICA FINALE (Ordinata per Response Time MAPE)")
    print("=" * 70)

    # ordina i risultati dal migliore (RT più basso) al peggiore
    results.sort(key=lambda x: x['RT_MAPE'])

    report_lines = []
    header = f"{'Pos':<5} | {'Features di Input':<35} | {'RT MAPE':<12} | {'U RMSE':<10}"
    print(header)
    print("-" * 70)
    report_lines.append(header)
    report_lines.append("-" * 70)

    for i, res in enumerate(results):
        line = f"{i + 1:<5} | {res['features']:<35} | {res['RT_MAPE']:>7.2f}%    | {res['U_RMSE']:>6.3f}"
        print(line)
        report_lines.append(line)

    out_file = os.path.join(out_dir, "Feature_Leaderboard.txt")
    with open(out_file, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n[OK] Leaderboard salvata in: {out_file}")


if __name__ == "__main__":
    main()