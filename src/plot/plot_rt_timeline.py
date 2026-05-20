import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 plot_rt_timeline.py <dir_esperimento>")
        sys.exit(1)

    base_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.join(base_dir, "rt_timeline_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ricerca row
    row_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("row_")],
                      key=lambda x: int(x.split('_')[1]))

    print(f"\nGenerazione Timeline per {len(row_dirs)} righe di carico...")

    for row_name in row_dirs:
        row_path = os.path.join(base_dir, row_name)
        jtl_files = [f for f in os.listdir(row_path) if f.endswith(".jtl")]

        if not jtl_files:
            continue

        jtl_path = os.path.join(row_path, jtl_files[0])

        try:
            df = pd.read_csv(jtl_path)
        except Exception as e:
            print(f"Errore lettura {jtl_path}: {e}")
            continue

        # ricerca funzioni nel jtl
        func_labels = df['label'].unique()

        for label in func_labels:
            if not label.startswith("Invoke_func_"):
                continue

            func_id = label.split("_")[-1]
            df_func = df[df['label'] == label].copy()

            if df_func.empty:
                continue

            # Calcolo tempo relativo in secondi dall'inizio del test
            min_time = df_func['timeStamp'].min()
            df_func['relative_time_sec'] = (df_func['timeStamp'] - min_time) / 1000.0
            df_func['rt_sec'] = df_func['elapsed'] / 1000.0

            # separazione successi e fallimenti
            successes = df_func[df_func['success'] == True]
            failures = df_func[df_func['success'] == False]

            plt.figure(figsize=(14, 6))

            # plot dei successi (verde)
            plt.scatter(successes['relative_time_sec'], successes['rt_sec'],
                        c='green', alpha=0.5, s=15, label='Successo (200 OK)')

            # plot dei fallimenti (rosso)
            if not failures.empty:
                plt.scatter(failures['relative_time_sec'], failures['rt_sec'],
                            c='red', marker='x', alpha=0.7, s=30, label='Fallito / Timeout')

            plt.title(f"Timeline RT - {row_name.upper()} - Funzione {func_id}", fontsize=14, fontweight='bold')
            plt.xlabel("Tempo dall'inizio del test (secondi)", fontsize=12)
            plt.ylabel("Response Time (secondi)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=12)
            plt.tight_layout()

            plot_name = f"timeline_{row_name}_func_{func_id}.png"
            plt.savefig(os.path.join(out_dir, plot_name), dpi=200)
            plt.close()

    print(f"\n[SUCCESSO] Grafici timeline salvati in:\n -> {out_dir}")


if __name__ == "__main__":
    main()