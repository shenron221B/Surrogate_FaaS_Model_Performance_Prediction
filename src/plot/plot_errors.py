import sys
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    if len(sys.argv) < 2:
        print("Errore: Passare il path del file boxplot_data.json")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"File JSON non trovato: {json_path}")
        sys.exit(1)

    ml_dir = os.path.dirname(json_path)
    base_dir = os.path.dirname(ml_dir)
    boxplot_dir = os.path.join(base_dir, "boxplot")
    os.makedirs(boxplot_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    sns.set_theme(style="whitegrid")

    # cicla dinamicamente su tutti i target salvati nel JSON
    for target_name, target_info in data.items():
        err_type = target_info["type"]  # "MAPE" o "RMSE"
        models_dict = target_info["models"]

        labels = list(models_dict.keys())
        values = list(models_dict.values())

        plt.figure(figsize=(10, 6))

        palette = "Set3" if err_type == "MAPE" else "Set2"

        sns.boxplot(data=values, palette=palette, showfliers=False)
        plt.xticks(range(len(labels)), labels, rotation=45)

        plt.title(f"Confronto Errori ({err_type}) - Target: {target_name}", fontsize=14, pad=15)
        plt.ylabel(f"{err_type} {'(%)' if err_type == 'MAPE' else ''}")
        plt.xlabel("Modelli")
        plt.tight_layout()

        save_path = os.path.join(boxplot_dir, f"boxplot_{target_name}_{err_type}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    print(f"[SUCCESSO] {len(data.keys())} Boxplot generati nella cartella: {boxplot_dir}")


if __name__ == "__main__":
    main()