import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def extract_raw_data(npz_path):
    data = np.load(npz_path)

    keys_to_check = ['X', 'Success', 'RT', 'Queue', 'U', 'Cold']
    min_len = min([data[k].shape[0] for k in keys_to_check if k in data.files])

    rt_raw = data['RT'][:min_len].flatten()
    queue_raw = data['Queue'][:min_len].flatten()
    u_raw = data['U'][:min_len].flatten() * 100
    cold_raw = data['Cold'][:min_len].flatten() * 100

    return rt_raw, queue_raw, u_raw, cold_raw


def compute_statistics(array):
    return {
        "Mean": float(np.mean(array)),
        "Median": float(np.median(array)),
        "Std_Dev": float(np.std(array))
    }


def analyze_scenario(scenario_dir, scenario_name, output_dir):
    print(f"\n[INFO] Analisi dello scenario: {scenario_name}")

    rep_dirs = glob.glob(os.path.join(scenario_dir, "*_rep*"))
    rep_dirs.sort(key=lambda x: int(x.split('_rep')[-1]))

    if not rep_dirs:
        print(f"[ATTENZIONE] Nessuna ripetizione trovata in {scenario_dir}")
        return

    print(f" -> Trovate {len(rep_dirs)} ripetizioni.")

    all_rt, all_queue, all_u, all_cold = [], [], [], []
    json_stats = {}

    for rep_path in rep_dirs:
        rep_name = "Rep " + rep_path.split('_rep')[-1]
        dataset_path = os.path.join(rep_path, "dataset.npz")

        if not os.path.exists(dataset_path):
            continue

        rt_raw, queue_raw, u_raw, cold_raw = extract_raw_data(dataset_path)

        all_rt.extend([(val, rep_name) for val in rt_raw])
        all_queue.extend([(val, rep_name) for val in queue_raw])
        all_u.extend([(val, rep_name) for val in u_raw])
        all_cold.extend([(val, rep_name) for val in cold_raw])

        json_stats[rep_name] = {
            "Response_Time_sec": compute_statistics(rt_raw),
            "Queue_Time_sec": compute_statistics(queue_raw),
            "Utility_Perc": compute_statistics(u_raw),
            "Cold_Start_Perc": compute_statistics(cold_raw)
        }

    df_rt = pd.DataFrame(all_rt, columns=['Valore', 'Ripetizione'])
    df_queue = pd.DataFrame(all_queue, columns=['Valore', 'Ripetizione'])
    df_u = pd.DataFrame(all_u, columns=['Valore', 'Ripetizione'])
    df_cold = pd.DataFrame(all_cold, columns=['Valore', 'Ripetizione'])

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Analisi Variabilità Sistema Reale - Scenario {scenario_name}", fontsize=20, fontweight='bold')

    palette = sns.color_palette("Set2")

    # 1. Response Time
    sns.boxplot(data=df_rt, x='Ripetizione', y='Valore', ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title("Response Time", fontsize=18)
    axes[0, 0].set_ylabel("Secondi (s)", fontsize=16)
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis='both', labelsize=14)

    # 2. Queueing Time
    sns.boxplot(data=df_queue, x='Ripetizione', y='Valore', ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title("Queueing Time", fontsize=18)
    axes[0, 1].set_ylabel("Secondi (s)", fontsize=16)
    axes[0, 1].set_xlabel("")
    axes[0, 1].tick_params(axis='both', labelsize=14)

    # 3. Utility
    sns.boxplot(data=df_u, x='Ripetizione', y='Valore', ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title("Utility (Rispetto Deadline)", fontsize=18)
    axes[1, 0].set_ylabel("Probabilità (%)", fontsize=16)
    axes[1, 0].set_xlabel("")
    axes[1, 0].tick_params(axis='both', labelsize=14)

    # 4. Cold Start
    sns.boxplot(data=df_cold, x='Ripetizione', y='Valore', ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title("Tasso di Cold Start", fontsize=18)
    axes[1, 1].set_ylabel("Probabilità (%)", fontsize=16)
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, f"variability_plot_{scenario_name}.png")
    json_path = os.path.join(output_dir, f"variability_stats_{scenario_name}.json")

    plt.savefig(plot_path, dpi=300)
    plt.close()

    with open(json_path, 'w') as f:
        json.dump(json_stats, f, indent=4)


def main():
    base_dir = "/root/tesi_project/serverledge/results/variability_test"

    if not os.path.exists(base_dir):
        print(f"[ERRORE] La cartella {base_dir} non esiste.")
        sys.exit(1)

    scenarios = {
        "2_Funzioni_Coda_5": os.path.join(base_dir, "2f2GBpm_qlen5_8core_256"),
        "5_Funzioni_Coda_5": os.path.join(base_dir, "5f2GBpm_qlen5_8core_256")
    }

    print("=" * 60)
    print(" AVVIO ANALISI DI VARIABILITA' SPERIMENTALE")
    print("=" * 60)

    for scenario_name, scenario_path in scenarios.items():
        if os.path.exists(scenario_path):
            analyze_scenario(scenario_path, scenario_name, base_dir)

    print("\n[INFO] Analisi completata!")


if __name__ == "__main__":
    main()
