import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import json


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 plot_warm_cold_timeline.py <dir_esperimento>")
        sys.exit(1)

    base_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.join(base_dir, "warm_cold_timeline_analysis_new_VM")
    os.makedirs(out_dir, exist_ok=True)

    row_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith("row_")],
                      key=lambda x: int(x.split('_')[1]))

    print(f"\nGenerazione Timeline Warm/Cold per {len(row_dirs)} righe di carico...")

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

        func_labels = df['label'].unique()

        for label in func_labels:
            if not label.startswith("Invoke_func_"):
                continue

            func_id = label.split("_")[-1]
            txt_file = os.path.join(row_path, f"http_responses_func{func_id}.txt")

            if not os.path.exists(txt_file):
                continue

            df_func = df[(df['label'] == label) & (df['success'] == True)].copy()
            if df_func.empty:
                continue

            min_time = df_func['timeStamp'].min()
            df_func['relative_time_sec'] = (df_func['timeStamp'] - min_time) / 1000.0
            jtl_times = df_func['relative_time_sec'].tolist()
            jtl_rts = (df_func['elapsed'] / 1000.0).tolist()

            is_warm_list = []
            with open(txt_file, 'r') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if not line: continue
                        data = json.loads(line)
                        if "Success" in data and not data["Success"]: continue

                        is_warm_list.append(data.get("IsWarmStart", True))
                    except json.JSONDecodeError:
                        continue

            min_len = min(len(jtl_times), len(is_warm_list))

            warm_x, warm_y = [], []
            cold_x, cold_y = [], []

            for i in range(min_len):
                if is_warm_list[i]:
                    warm_x.append(jtl_times[i])
                    warm_y.append(jtl_rts[i])
                else:
                    cold_x.append(jtl_times[i])
                    cold_y.append(jtl_rts[i])

            plt.figure(figsize=(14, 6))

            bar_width = 0.5

            if warm_x:
                plt.bar(warm_x, warm_y, width=bar_width, color='blue', alpha=0.7, label='Warm Start (Blu)')
            if cold_x:
                plt.bar(cold_x, cold_y, width=bar_width, color='orange', alpha=0.9, label='Cold Start (Arancione)')

            plt.title(f"Timeline Warm/Cold RT - {row_name.upper()} - Funzione {func_id}", fontsize=14,
                      fontweight='bold')
            plt.xlabel("Tempo dall'inizio del test (secondi)", fontsize=12)
            plt.ylabel("Response Time (secondi)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend(fontsize=12)
            plt.tight_layout()

            plot_name = f"warm_cold_timeline_{row_name}_func_{func_id}.png"
            plt.savefig(os.path.join(out_dir, plot_name), dpi=200)
            plt.close()

    print(f"\n[SUCCESSO] Grafici timeline Warm/Cold salvati in:\n -> {out_dir}")


if __name__ == "__main__":
    main()