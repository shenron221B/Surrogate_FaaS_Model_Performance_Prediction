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
    print("Uso: python3 plot_gamma_theory.py <path_cartella_principale>")
    print("Esempio: python3 plot_gamma_theory.py serverledge/results/1f_2GBpm/q_len_5/256/x15")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    conf_file = os.path.join(base_dir, "simulator-conf.yml")

    if not os.path.exists(conf_file):
        print(f"Errore: File {conf_file} non trovato.")
        sys.exit(1)

    # lettura dei parametri del simulatore dal file YAML
    with open(conf_file, 'r') as f:
        config = yaml.safe_load(f)

    means = config.get("serv_time_duration", [])
    cvs = config.get("serv_time_cvs", [])
    num_funcs = len(means)

    print(f"\nGenerazione Plot Gamma Teorico vs Reale per {num_funcs} funzioni...")

    for i in range(num_funcs):
        func_id = i + 1
        mu = means[i]
        cv = cvs[i]

        if cv == 0 or mu == 0:
            print(f"  [F{func_id}] CV o Mu nulli. Impossibile calcolare Gamma.")
            continue

        # formule matematiche per la conversione
        shape_k = 1.0 / (cv ** 2)
        scale_theta = mu * (cv ** 2)

        # raccolta di tutti i dati empirici per questa funzione da tutte le row
        empirical_durations = []
        txt_files = glob.glob(os.path.join(base_dir, "row_*", f"http_responses_func{func_id}.txt"))

        for txt in txt_files:
            try:
                with open(txt, 'r') as f:
                    for line in f:
                        if line.startswith("{"):
                            data = json.loads(line.strip())
                            if data.get('Success') is True:
                                empirical_durations.append(data.get('Duration', 0))
            except:
                pass

        if not empirical_durations:
            print(f"  [F{func_id}] Nessun dato empirico trovato.")
            continue

        # rimozione outlier estremi per visualizzazione pulita
        p99 = np.percentile(empirical_durations, 99)
        clean_durations = [d for d in empirical_durations if d <= p99]

        # plot
        plt.figure(figsize=(9, 5))

        # istogramma empirico
        plt.hist(clean_durations, bins=40, density=True, alpha=0.5, color='royalblue', edgecolor='black',
                 label=f'Dati Reali (F{func_id})')

        # curva teorica usata dal simulatore Go
        x_val = np.linspace(min(clean_durations) * 0.9, max(clean_durations) * 1.1, 500)
        y_theoretical = stats.gamma.pdf(x_val, a=shape_k, scale=scale_theta)

        plt.plot(x_val, y_theoretical, 'r-', lw=2.5,
                 label=f'Curva Teorica Go\n(Shape k={shape_k:.1f}, Scale \u03B8={scale_theta:.5f})')

        plt.title(f"Distribuzione Tempo di Servizio F{func_id}\n\u03BC={mu:.4f}s | CV={cv:.4f}")
        plt.xlabel("Tempo (Secondi)")
        plt.ylabel("Densità di Probabilità")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # salvataggio nella cartella dell'esperimento
        out_plot = os.path.join(base_dir, f"gamma_theory_vs_empirical_func{func_id}.png")
        plt.savefig(out_plot, dpi=200)
        plt.close()

        print(f"  [OK] Salvato: {os.path.basename(out_plot)}")

    print("\n[SUCCESSO] Plot generati correttamente!")


if __name__ == "__main__":
    main()