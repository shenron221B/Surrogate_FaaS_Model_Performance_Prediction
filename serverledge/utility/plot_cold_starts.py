import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 plot_cold_starts.py <dir_esperimento>")
        sys.exit(1)

    base_dir = os.path.abspath(sys.argv[1])
    matrix_file = os.path.join(base_dir, "X_matrix_used.txt")

    if not os.path.exists(matrix_file):
        print(f"[ERRORE] File matrice non trovato: {matrix_file}")
        sys.exit(1)

    loads = np.loadtxt(matrix_file, ndmin=2)
    num_rows, num_funcs = loads.shape

    out_dir = os.path.join(base_dir, "cold_start_analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nAnalisi in corso su {num_rows} righe per {num_funcs} funzioni...")

    x_labels_combined = []
    cold_starts_absolute = {f: [] for f in range(1, num_funcs + 1)}

    for func_idx in range(num_funcs):
        func_id = func_idx + 1
        x_labels = []
        cold_rates = []
        avg_inits = []
        std_inits = []

        for row_idx in range(num_rows):
            row_dir = os.path.join(base_dir, f"row_{row_idx + 1}")
            load_val = loads[row_idx, func_idx]

            x_labels.append(f"R{row_idx + 1}\n({load_val:.2f})")
            log_file = os.path.join(row_dir, f"http_responses_func{func_id}.txt")

            cold_count = 0
            total_count = 0
            inits = []

            if func_idx == 0:
                row_total_reqs = 0
                row_completed_reqs = 0
                loads_str = "[" + ", ".join([f"{loads[row_idx, i]:.2f}" for i in range(num_funcs)]) + "]"

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            line = line.strip()
                            if not line: continue
                            data = json.loads(line)

                            if func_idx == 0:
                                pass

                            if "Success" in data and not data["Success"]:
                                continue

                            is_warm = data.get("IsWarmStart", True)
                            total_count += 1

                            if not is_warm:
                                cold_count += 1
                                q_t = data.get("QueueingTime", 0)
                                i_t = data.get("InitTime", 0)
                                inits.append(max(0, i_t - q_t))
                        except json.JSONDecodeError:
                            continue

            rate = (cold_count / total_count * 100) if total_count > 0 else 0
            avg_i = np.mean(inits) if inits else 0
            std_i = np.std(inits) if len(inits) > 1 else 0

            cold_rates.append(rate)
            avg_inits.append(avg_i)
            std_inits.append(std_i)

        # PLOT 1: Cold Start Rate (%)
        plt.figure(figsize=(12, 6))
        plt.bar(x_labels, cold_rates, color='skyblue', edgecolor='black')
        plt.title(f"Funzione {func_id} - Cold Start Rate per Riga/Carico", fontsize=14, fontweight='bold')
        plt.ylabel("Cold Start Rate (%)", fontsize=12)
        plt.xlabel("Row / Carico (req/s)", fontsize=12)
        plt.ylim(0, 105)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"func_{func_id}_cold_rate.png"), dpi=300)
        plt.close()

        # PLOT 2: Init Time (s)
        plt.figure(figsize=(12, 6))
        plt.bar(x_labels, avg_inits, yerr=std_inits, color='coral', edgecolor='black', capsize=5, alpha=0.9)
        plt.title(f"Funzione {func_id} - Init Time Medio per Riga/Carico", fontsize=14, fontweight='bold')
        plt.ylabel("Init Time (secondi)", fontsize=12)
        plt.xlabel("Row / Carico (req/s)", fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"func_{func_id}_init_time.png"), dpi=300)
        plt.close()

    for row_idx in range(num_rows):
        row_dir = os.path.join(base_dir, f"row_{row_idx + 1}")
        row_total_reqs = 0
        row_completed_reqs = 0
        loads_str = "[" + ", ".join([f"{loads[row_idx, i]:.2f}" for i in range(num_funcs)]) + "]"

        for func_idx in range(num_funcs):
            func_id = func_idx + 1
            log_file = os.path.join(row_dir, f"http_responses_func{func_id}.txt")
            cold_count = 0

            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            line = line.strip()
                            if not line: continue
                            data = json.loads(line)

                            row_total_reqs += 1
                            if data.get("Success", False):
                                row_completed_reqs += 1

                            if not data.get("IsWarmStart", True):
                                cold_count += 1
                        except json.JSONDecodeError:
                            continue
            cold_starts_absolute[func_id].append(cold_count)

        label = f"{loads_str}\n{row_completed_reqs}/{row_total_reqs}"
        x_labels_combined.append(label)

    # PLOT 3: Cold Starts Assoluti
    plt.figure(figsize=(15, 7))
    x = np.arange(num_rows)
    width = 0.8 / num_funcs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for func_idx in range(num_funcs):
        func_id = func_idx + 1
        offset = (func_idx - num_funcs / 2 + 0.5) * width
        plt.bar(x + offset, cold_starts_absolute[func_id], width,
                label=f'Funzione {func_id}',
                color=colors[func_idx % len(colors)], edgecolor='black')

    plt.title("Numero Assoluto di Cold Start per Riga di Carico", fontsize=15, fontweight='bold')
    plt.ylabel("Numero Cold Start", fontsize=12)
    plt.xlabel("Carichi [req/s] \n (Completate / Loggate)", fontsize=12)
    plt.xticks(x, x_labels_combined, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, "combined_cold_starts_absolute.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[SUCCESSO] Grafici salvati nella cartella:\n -> {out_dir}")


if __name__ == "__main__":
    main()