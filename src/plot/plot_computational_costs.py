import os
import sys
import glob
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def print_usage():
    print("Uso: python3 plot_computational_costs.py <cartella_results_base>")
    print("Esempio: python3 plot_computational_costs.py /root/tesi_project/serverledge/results")
    sys.exit(1)


def extract_num_functions(folder_name):
    match = re.search(r'(\d+)f', folder_name)
    if match:
        return int(match.group(1))
    return None


def get_model_order(model_name):
    name = model_name.lower()
    if "grosof" in name or "msj" in name: return 0
    if "m/m" in name or "mmck" in name: return 1
    if "kaufman" in name: return 2
    if "simulat" in name: return 3
    if "ml" in name or "fine" in name: return 4
    return 5


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    search_pattern = os.path.join(base_dir, "**", "execution_times.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print(f"[ERRORE] Nessun file 'execution_times.json' trovato.")
        sys.exit(1)

    print(f"[INFO] Trovati {len(json_files)} file di log dei tempi. Estrazione in corso...")
    records = []

    for jpath in json_files:
        folder_name = os.path.basename(os.path.dirname(jpath))
        num_f = extract_num_functions(folder_name)

        if num_f is None:
            continue

        with open(jpath, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        for model_name, info in data.items():
            records.append({
                "Num_Funzioni": f"{num_f} Funzioni",
                "Modello": model_name,
                "Tempo_Secondi": info.get("total_time_seconds", 0.0)
            })

    if not records:
        print("[ERRORE] Nessun dato valido estratto.")
        sys.exit(1)

    df = pd.DataFrame(records)
    df_mean = df.groupby(["Num_Funzioni", "Modello"])["Tempo_Secondi"].mean().reset_index()

    out_dir = os.path.join(base_dir, "computational_costs")
    os.makedirs(out_dir, exist_ok=True)

    # tabella csv
    df_pivot = df_mean.pivot(index="Modello", columns="Num_Funzioni", values="Tempo_Secondi")
    df_pivot_formatted = df_pivot.applymap(lambda x: f"{x:.4f}" if x < 1 else f"{x:,.2f}")
    csv_path = os.path.join(out_dir, "tabella_costi_computazionali.csv")
    df_pivot_formatted.to_csv(csv_path)

    # ordinamento delle barre per modello
    modelli_unici = df_mean["Modello"].unique().tolist()
    ordine_esatto = sorted(modelli_unici, key=get_model_order)

    # grafico scala log
    plt.figure(figsize=(12, 7))
    palette = sns.color_palette("Set2", len(ordine_esatto))

    ax = sns.barplot(
        data=df_mean,
        x="Num_Funzioni",
        y="Tempo_Secondi",
        hue="Modello",
        hue_order=ordine_esatto,
        palette=palette,
        edgecolor=".2"
    )

    ax.set_yscale("log")
    ax.set_xlabel("Complessità del Sistema", fontsize=16)
    ax.set_ylabel("Tempo Computazionale (Secondi) - Scala Log", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7, which="both")

    plt.tight_layout()
    plot_path = os.path.join(out_dir, "grafico_costi_computazionali.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[OK] Grafico salvato in: {plot_path}\n")

if __name__ == "__main__":
    main()
