import sys
import os
import glob
import json
import matplotlib.pyplot as plt
import seaborn as sns

def print_usage():
    print("Uso: python3 plot_model_errors.py <numero_funzioni> <nome_matrice>")
    print("Esempio: python3 plot_model_errors.py 2 2f_mat_hash_poisson600_matrix1")
    sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print_usage()

    num_f = sys.argv[1]
    mat_name = sys.argv[2]
    base_results = "/root/tesi_project/serverledge/results"
    qlens = [0, 5, 15]

    sns.set_theme(style="whitegrid")

    for q in qlens:
        # costruzione path
        dir_name = f"{num_f}f2GBpm_qlen{q}_8core_256"
        target_dir = os.path.join(base_results, dir_name, mat_name)

        if not os.path.exists(target_dir):
            print(f"[SKIP] Cartella non trovata: {target_dir}")
            continue

        # cerca il JSON di Fine Tuning
        json_files = glob.glob(os.path.join(target_dir, "fine_tuning_*", "boxplot_data.json"))
        if not json_files:
            json_files = glob.glob(os.path.join(target_dir, "fine_tuning_*", "rt_utility", "boxplot_data.json"))

        if not json_files:
            print(f"[SKIP] Nessun boxplot_data.json in {target_dir}")
            continue

        json_path = json_files[0]
        
        with open(json_path, "r") as f:
            data = json.load(f)

        # creazione cartella output
        out_dir = os.path.join(target_dir, "fine_tuning_boxplot")
        os.makedirs(out_dir, exist_ok=True)

        # inizializza Griglia 2x2
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        axes = axes.flatten()

        print(f"\n[INFO] Generazione Griglia 2x2 per {dir_name}...")

        for idx, (target_name, target_info) in enumerate(data.items()):
            if idx >= 4: 
                break

            err_type = target_info["type"]  # "MAPE" o "RMSE"
            models_dict = target_info["models"]

            labels = list(models_dict.keys())
            values = list(models_dict.values())

            ax = axes[idx]
            palette = "Set3" if err_type == "MAPE" else "Set2"

            sns.boxplot(data=values, ax=ax, palette=palette, showfliers=False, width=0.6)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=20)

            ax.set_title(f"Target: {target_name}", fontsize=24, pad=15, fontweight='normal')
            
            ax.set_ylabel(f"{err_type} {'(%)' if err_type == 'MAPE' else ''}", fontsize=18)
            ax.tick_params(axis='y', labelsize=16)

        for i in range(len(data.items()), 4):
            fig.delaxes(axes[i])

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"combined_boxplots_qlen{q}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f" -> [OK] Salvato in: {save_path}")

    print("\n[OPERAZIONE COMPLETATA]")

if __name__ == "__main__":
    main()
