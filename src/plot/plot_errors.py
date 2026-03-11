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

    # creazione cartella boxplot
    ml_dir = os.path.dirname(json_path)
    base_dir = os.path.dirname(ml_dir)
    boxplot_dir = os.path.join(base_dir, "boxplot")
    os.makedirs(boxplot_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    # stile grafico
    sns.set_theme(style="whitegrid")

    # 1. PLOT MAPE (Response Time)
    plt.figure(figsize=(10, 6))
    labels_rt = list(data["RT_MAPE"].keys())
    values_rt = list(data["RT_MAPE"].values())

    sns.boxplot(data=values_rt, palette="Set3", showfliers=False)
    plt.xticks(range(len(labels_rt)), labels_rt, rotation=45)
    plt.title("Confronto Errori (MAPE) - Tempo di Risposta (RT)", fontsize=14, pad=15)
    plt.ylabel("MAPE (%)")
    plt.xlabel("Modelli")
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, "boxplot_RT_MAPE.png"), dpi=300)
    plt.close()

    # 2. PLOT RMSE (Utility)
    plt.figure(figsize=(10, 6))
    labels_u = list(data["U_RMSE"].keys())
    values_u = list(data["U_RMSE"].values())

    sns.boxplot(data=values_u, palette="Set2", showfliers=False)
    plt.xticks(range(len(labels_u)), labels_u, rotation=45)
    plt.title("Confronto Errori (RMSE) - Utilità", fontsize=14, pad=15)
    plt.ylabel("RMSE")
    plt.xlabel("Modelli")
    plt.tight_layout()
    plt.savefig(os.path.join(boxplot_dir, "boxplot_Utility_RMSE.png"), dpi=300)
    plt.close()

    print(f"[SUCCESSO] Boxplot generati nella cartella: {boxplot_dir}")


if __name__ == "__main__":
    main()