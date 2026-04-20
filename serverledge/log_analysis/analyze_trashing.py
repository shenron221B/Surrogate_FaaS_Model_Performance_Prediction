import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def print_usage():
    print("Uso: python3 analyze_thrashing.py <dir_esperimento> <num_ultime_righe>")
    print("Esempio: python3 analyze_thrashing.py /root/tesi/results/VM1_2f 5")
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    try:
        num_last_rows = int(sys.argv[2])
    except ValueError:
        print("[ERRORE] Il numero di righe deve essere un intero.")
        sys.exit(1)

    log_file = os.path.join(base_dir, "serverledge_full_log.txt")
    if not os.path.exists(log_file):
        print(f"[ERRORE] File {log_file} non trovato in {base_dir}.")
        sys.exit(1)

    print(f"Analisi del log: {log_file} ...\n")

    # contatori per riga
    rows_data = []
    current_row = None

    with open(log_file, "r") as f:
        for line in f:
            # ogni volta che Serverledge restarta è una nuova riga dell'esperimento
            if "Scheduler started." in line:
                if current_row is not None:
                    rows_data.append(current_row)
                current_row = {
                    "thrashing": 0,
                    "queue_full": 0,
                    "out_of_res": 0,
                    "drop": 0
                }

            # controllo anomalie dentro una riga
            if current_row is not None:
                if "[THRASHING-ALERT]" in line:
                    current_row["thrashing"] += 1
                elif "[QUEUE-FULL]" in line:
                    current_row["queue_full"] += 1
                elif "[OUT-OF-RES]" in line:
                    current_row["out_of_res"] += 1
                elif "[DROP-FATAL]" in line:
                    current_row["drop"] += 1

    if current_row is not None:
        rows_data.append(current_row)

    if not rows_data:
        print("[AVVISO] Nessun avvio di Serverledge trovato nel log.")
        return

    # filtro le ultime N righe
    if len(rows_data) < num_last_rows:
        print(f"[AVVISO] Hai richiesto {num_last_rows} righe, ma il log ne contiene solo {len(rows_data)}.")
        target_rows = rows_data
    else:
        target_rows = rows_data[-num_last_rows:]

    # stampa i risultati a terminale
    print(
        f"{'ROW':<5} | {'THRASHING (Evictions)':<25} | {'QUEUE FULL':<15} | {'OUT OF RES (Cold Fails)':<25} | {'DROP FATAL':<15}")
    print("-" * 90)

    thrashing_counts = []
    queue_counts = []
    out_of_res_counts = []
    drop_counts = []
    labels = []

    for i, data in enumerate(target_rows):
        # Indice relativo alle righe analizzate
        row_idx = i + 1
        labels.append(f"R{row_idx}")

        thrashing_counts.append(data['thrashing'])
        queue_counts.append(data['queue_full'])
        out_of_res_counts.append(data['out_of_res'])
        drop_counts.append(data['drop'])

        print(
            f"{row_idx:<5} | {data['thrashing']:<25} | {data['queue_full']:<15} | {data['out_of_res']:<25} | {data['drop']:<15}")

    # generazione grafico a barre affiancate
    x = np.arange(len(labels))
    width = 0.2  # larghezza barre

    fig, ax = plt.subplots(figsize=(14, 7))

    rects1 = ax.bar(x - 1.5 * width, thrashing_counts, width, label='Thrashing (Evictions)', color='crimson',
                    edgecolor='black')
    rects2 = ax.bar(x - 0.5 * width, queue_counts, width, label='Queue Full', color='orange', edgecolor='black')
    rects3 = ax.bar(x + 0.5 * width, out_of_res_counts, width, label='Out of Resources', color='purple',
                    edgecolor='black')
    rects4 = ax.bar(x + 1.5 * width, drop_counts, width, label='Drop Fatal (429)', color='black', edgecolor='white')

    ax.set_title(f"Analisi Anomalie Serverledge (Ultime {len(labels)} righe di carico)", fontsize=15, fontweight='bold')
    ax.set_xlabel("Righe dell'Esperimento", fontsize=12)
    ax.set_ylabel("Numero di Eventi", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # funzione helper per aggiungere le label sopra le barre
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 punti offset verticale
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, rotation=90)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    out_png = os.path.join(base_dir, "anomalies_analysis_plot.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print(f"\n[SUCCESSO] Grafico salvato in: {out_png}")


if __name__ == "__main__":
    main()