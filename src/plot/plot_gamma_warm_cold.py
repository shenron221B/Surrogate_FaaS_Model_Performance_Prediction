import sys
import os
import glob
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.switch_backend('Agg')


def print_usage():
    print("Uso: python3 plot_gamma_warm_cold.py <path_cartella_principale>")
    print("Esempio: python3 plot_gamma_warm_cold.py serverledge/results/1f_2GBpm/q_len_5/256/x15")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    conf_file = os.path.join(base_dir, "simulator-conf.yml")

    if not os.path.exists(conf_file):
        print(f"[ERRORE] File {conf_file} non trovato.")
        sys.exit(1)

    # lettura parametri simulatore
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)

    means = config.get("serv_time_duration", [])
    cvs = config.get("serv_time_cvs", [])
    num_funcs = len(means)

    print(f"\nGenerazione Plot Gamma Teorico vs Warm/Cold per {num_funcs} funzioni...")

    for i in range(num_funcs):
        func_id = i + 1
        mu = means[i]
        cv = cvs[i]

        if cv == 0 or mu == 0:
            print(f"  [F{func_id}] CV o Mu nulli. Salto.")
            continue

        shape_k = 1.0 / (cv ** 2)
        scale_theta = mu * (cv ** 2)

        warm_durations = []
        cold_durations = []

        txt_files = glob.glob(os.path.join(base_dir, "row_*", f"http_responses_func{func_id}.txt"))

        for txt in txt_files:
            try:
                with open(txt, 'r') as f:
                    for line in f:
                        if line.startswith("{"):
                            data = json.loads(line.strip())
                            if data.get('Success') is True:
                                duration = data.get('Duration', 0)
                                is_warm = data.get('IsWarmStart', False)

                                if is_warm:
                                    warm_durations.append(duration)
                                else:
                                    cold_durations.append(duration)
            except:
                pass

        all_durations = warm_durations + cold_durations
        if not all_durations:
            print(f"  [F{func_id}] Nessun dato empirico trovato.")
            continue

        # taglio al 99° percentile globale
        p99 = np.percentile(all_durations, 99)
        clean_warm = [d for d in warm_durations if d <= p99]
        clean_cold = [d for d in cold_durations if d <= p99]
        clean_all = clean_warm + clean_cold

        perc_warm = (len(clean_warm) / len(clean_all)) * 100
        perc_cold = (len(clean_cold) / len(clean_all)) * 100

        # --- PLOT ---
        plt.figure(figsize=(10, 6))

        bins = np.histogram_bin_edges(clean_all, bins=40)

        # istogramma stacked normalizzato
        plt.hist([clean_warm, clean_cold], bins=bins, density=True, stacked=True,
                 color=['#8DA0CB', '#FC8D62'], edgecolor='black', alpha=0.8,
                 label=[f'Warm Starts ({perc_warm:.1f}%)', f'Cold Starts ({perc_cold:.1f}%)'])

        # curva teorica Go
        x_val = np.linspace(min(clean_all) * 0.9, max(clean_all) * 1.1, 500)
        y_theoretical = stats.gamma.pdf(x_val, a=shape_k, scale=scale_theta)

        plt.plot(x_val, y_theoretical, 'r-', lw=2.5,
                 label=f'Curva Teorica Go\n(Shape k={shape_k:.1f}, Scale \u03B8={scale_theta:.5f})')

        plt.title(f"Distribuzione Tempo di Servizio F{func_id} (Warm vs Cold)\n\u03BC={mu:.4f}s | CV={cv:.4f}")
        plt.xlabel("Tempo (Secondi)")
        plt.ylabel("Densità di Probabilità")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        out_plot = os.path.join(base_dir, f"gamma_warm_cold_func{func_id}.png")
        plt.savefig(out_plot, dpi=200)
        plt.close()

        print(f"  [OK] Salvato: {os.path.basename(out_plot)}")

    print("\n[SUCCESSO] Plot generati correttamente!")


if __name__ == "__main__":
    main()