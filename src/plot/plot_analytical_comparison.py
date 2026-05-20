import sys
import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print_usage():
    print("Uso: python3 plot_analytical_comparison.py <dir_risultati>")
    sys.exit(1)

def find_best_ft(models_dict):
    best_ft_key = None
    best_mean = float('inf')
    for key, arr in models_dict.items():
        if key.startswith("FT_"):
            current_mean = np.mean(arr)
            if current_mean < best_mean:
                best_mean = current_mean
                best_ft_key = key
    return best_ft_key

def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    
    analyt_json = os.path.join(base_dir, "analytical_errors.json")
    msj_json = os.path.join(base_dir, "msj_errors.json")
    
    ml_files = glob.glob(os.path.join(base_dir, "fine_tuning_*", "boxplot_data.json"))
    if not ml_files:
        ml_files = glob.glob(os.path.join(base_dir, "fine_tuning_*", "rt_utility", "boxplot_data.json"))
    
    if not os.path.exists(analyt_json) or not os.path.exists(msj_json) or not ml_files:
        print("[ERRORE] File JSON mancanti.")
        sys.exit(1)

    ml_json = ml_files[0]
    
    with open(analyt_json, 'r') as f: data_an = json.load(f)
    with open(msj_json, 'r') as f: data_msj = json.load(f)
    with open(ml_json, 'r') as f: data_ml = json.load(f)

    print(f"\nGenerazione Master Plot in corso...")

    sns.set_theme(style="whitegrid")
    
    # ==========================================
    # GRAFICO 1: MAPE su Response Time (RT)
    # ==========================================
    best_ft_rt = find_best_ft(data_ml["RT"]["models"])
    
    rt_data = []
    for val in data_ml["RT"]["models"].get("Simulatore", []): rt_data.append({"Modello": "Simulatore", "Errore": val})
    for val in data_ml["RT"]["models"].get("ML_Sintetico", []): rt_data.append({"Modello": "ML Zero-Shot", "Errore": val})
    if best_ft_rt:
        for val in data_ml["RT"]["models"][best_ft_rt]: rt_data.append({"Modello": f"ML Fine-Tuned\n({best_ft_rt})", "Errore": val})
    for val in data_an["RT"].get("M/M/c/K", []): rt_data.append({"Modello": "M/M/c/K", "Errore": val})
    for val in data_an["RT"].get("Kaufman", []): rt_data.append({"Modello": "Kaufman", "Errore": val})
    for val in data_msj["RT"].get("Grosof (MSJ)", []): rt_data.append({"Modello": "Grosof (MSJ)", "Errore": val})

    df_rt = pd.DataFrame(rt_data)
    
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(data=df_rt, x="Modello", y="Errore", hue="Modello", palette="Set2", showfliers=False, width=0.6, legend=False)

    plt.title("Confronto Architetture - MAPE su Response Time (RT)", fontsize=18, pad=15)
    plt.ylabel("MAPE (%)", fontsize=16)
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plot_rt_path = os.path.join(base_dir, "analytical_comparison_RT_2fqlen0.png")
    plt.savefig(plot_rt_path, dpi=300)
    plt.close()

    # ==========================================
    # GRAFICO 2: RMSE su Utility (U)
    # ==========================================
    best_ft_u = find_best_ft(data_ml["U"]["models"])

    u_data = []
    for val in data_ml["U"]["models"].get("Simulatore", []): u_data.append({"Modello": "Simulatore", "Errore": val})
    for val in data_ml["U"]["models"].get("ML_Sintetico", []): u_data.append({"Modello": "ML Zero-Shot", "Errore": val})
    if best_ft_u:
        for val in data_ml["U"]["models"][best_ft_u]: u_data.append({"Modello": f"ML Fine-Tuned\n({best_ft_u})", "Errore": val})
    for val in data_an["U"].get("M/M/c/K", []): u_data.append({"Modello": "M/M/c/K", "Errore": val})
    for val in data_an["U"].get("Kaufman", []): u_data.append({"Modello": "Kaufman", "Errore": val})

    df_u = pd.DataFrame(u_data)
    
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(data=df_u, x="Modello", y="Errore", hue="Modello", palette="Set3", showfliers=False, width=0.6, legend=False)

    plt.title("Confronto Architetture - RMSE su Utility", fontsize=18, pad=15)
    plt.ylabel("RMSE (Probabilità)", fontsize=16)
    plt.xlabel("")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plot_u_path = os.path.join(base_dir, "analytical_comparison_U_2fqlen0.png")
    plt.savefig(plot_u_path, dpi=300)
    plt.close()

    print(f"\n[SUCCESSO] Grafici salvati.")

if __name__ == "__main__":
    main()
