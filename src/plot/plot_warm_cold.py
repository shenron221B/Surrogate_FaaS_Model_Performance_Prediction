import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 plot_warm_cold.py <Dir_Risultati> <Num_Funzioni>")
        sys.exit(1)

    RESULTS_DIR = sys.argv[1]
    FUNCTIONS = int(sys.argv[2])
    FUNC_OFFSET = 1

    if not os.path.exists(RESULTS_DIR):
        print(f"[ERRORE] Cartella {RESULTS_DIR} non trovata.")
        sys.exit(1)

    rows = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("row")], key=lambda x: int(x.split('_')[1]))

    for f_id in range(FUNC_OFFSET, FUNC_OFFSET + FUNCTIONS):
        warm_times = []
        cold_times = []

        print(f"\nAnalisi Funzione {f_id}...")

        for row in rows:
            row_path = os.path.join(RESULTS_DIR, row)
            file_path = os.path.join(row_path, f"http_responses_func{f_id}.txt")

            if not os.path.exists(file_path):
                continue

            with open(file_path, "r") as file:
                for line in file:
                    try:
                        data = json.loads(line.strip())
                        if "Success" in data and not data["Success"]:
                            continue

                        # estrazione del tempo
                        q_t = data.get("QueueingTime", 0)
                        init_time = data.get("InitTime", 0)
                        duration = data.get("Duration", 0)
                        is_warm = data.get("IsWarmStart", False)

                        if is_warm:
                            # Warm Start
                            val = duration + max(0, init_time - q_t)
                            warm_times.append(val)
                        else:
                            # Cold Start
                            val = duration
                            cold_times.append(val)

                    except json.JSONDecodeError:
                        continue

        # statistiche
        total_req = len(warm_times) + len(cold_times)
        if total_req == 0:
            print(f"Nessun dato per la funzione {f_id}")
            continue

        perc_warm = (len(warm_times) / total_req) * 100
        perc_cold = (len(cold_times) / total_req) * 100

        print(f"  Totale richieste : {total_req}")
        print(f"  Warm Starts      : {len(warm_times)} ({perc_warm:.1f}%) | Media: {np.mean(warm_times):.4f}s")
        print(f"  Cold Starts      : {len(cold_times)} ({perc_cold:.1f}%) | Media: {np.mean(cold_times):.4f}s")

        # plot istogramma
        plt.figure(figsize=(10, 6))

        all_times = warm_times + cold_times
        bins = np.histogram_bin_edges(all_times, bins=50)

        # plot impilato
        plt.hist([warm_times, cold_times], bins=bins, stacked=True,
                 color=['#8DA0CB', '#FC8D62'], edgecolor='black', alpha=0.8,
                 label=[f'Warm Starts ({perc_warm:.1f}%)', f'Cold Starts ({perc_cold:.1f}%)'])

        plt.title(f"Distribuzione Tempo di Servizio - F{f_id}\nVerifica Ipotesi Warm vs Cold")
        plt.xlabel("Tempo di Servizio (Secondi)")
        plt.ylabel("Numero di Richieste (Frequenza Assoluta)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # salvataggio
        out_img = os.path.join(RESULTS_DIR, f"dist_warm_cold_F{f_id}.png")
        plt.savefig(out_img, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Grafico salvato in: {out_img}")


if __name__ == "__main__":
    main()