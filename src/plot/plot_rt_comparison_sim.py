import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# Costanti
CHUNK_SIZE = 38  # Numero massimo di righe (scenari) per ogni grafico
FUNC_OFFSET = 1


def print_usage():
    print("Uso: python3 plot_rt_comparison.py <dir_esperimento>")
    print("Esempio: python3 plot_rt_comparison.py /root/.../poisson_600/2f_matrixmem_hashworker_matrix")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    sim_npz_file = os.path.join(base_dir, "dataset_simulated_exact.npz")
    matrix_file = os.path.join(base_dir, "X_matrix_used.txt")

    if not os.path.exists(sim_npz_file):
        print(f"[ERRORE] File simulatore non trovato: {sim_npz_file}")
        sys.exit(1)
    if not os.path.exists(matrix_file):
        print(f"[ERRORE] File matrice non trovato: {matrix_file}")
        sys.exit(1)

    # Caricamento dati
    sim_data = np.load(sim_npz_file)
    sim_rt_matrix = sim_data['RT']  # Tempi medi del simulatore

    loads = np.loadtxt(matrix_file, ndmin=2)
    num_rows, num_funcs = loads.shape

    out_dir = os.path.join(base_dir, "rt_comparison_analysis_0_matrix")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerazione grafici di confronto RT su {num_rows} righe per {num_funcs} funzioni...")

    # Estrazione dati reali dai .jtl
    real_mean_rt = np.zeros((num_rows, num_funcs))
    real_median_rt = np.zeros((num_rows, num_funcs))

    for row_idx in range(num_rows):
        row_dir = os.path.join(base_dir, f"row_{row_idx + 1}")
        jtl_files = [f for f in os.listdir(row_dir) if f.endswith(".jtl")]

        if not jtl_files:
            continue

        jtl_path = os.path.join(row_dir, jtl_files[0])
        try:
            df = pd.read_csv(jtl_path)
        except Exception as e:
            print(f"[WARN] Impossibile leggere {jtl_path}: {e}")
            continue

        for func_idx in range(num_funcs):
            func_id = func_idx + FUNC_OFFSET
            df_func = df[(df["label"] == f"Invoke_func_{func_id}") & (df["success"] == True)]

            if not df_func.empty:
                real_mean_rt[row_idx, func_idx] = df_func["elapsed"].mean() / 1000.0
                real_median_rt[row_idx, func_idx] = df_func["elapsed"].median() / 1000.0

    # Generazione dei plot divisi in chunk
    num_chunks = math.ceil(num_rows / CHUNK_SIZE)

    for func_idx in range(num_funcs):
        func_id = func_idx + FUNC_OFFSET

        for chunk in range(num_chunks):
            start_idx = chunk * CHUNK_SIZE
            end_idx = min((chunk + 1) * CHUNK_SIZE, num_rows)

            chunk_length = end_idx - start_idx
            x = np.arange(chunk_length)
            width = 0.35

            # Etichette asse X
            x_labels = []
            for r in range(start_idx, end_idx):
                load_val = loads[r, func_idx]
                x_labels.append(f"R{r + 1}\n({load_val:.2f})")

            sim_rt_chunk = sim_rt_matrix[start_idx:end_idx, func_idx]
            real_mean_chunk = real_mean_rt[start_idx:end_idx, func_idx]
            real_median_chunk = real_median_rt[start_idx:end_idx, func_idx]

            # ==========================================
            # PLOT 1: SIM MEAN vs REAL MEAN
            # ==========================================
            fig, ax = plt.subplots(figsize=(14, 6))

            rects1 = ax.bar(x - width / 2, sim_rt_chunk, width, label='Simulatore (Media)', color='#1f77b4',
                            edgecolor='black')
            rects2 = ax.bar(x + width / 2, real_mean_chunk, width, label='Serverledge (Media)', color='#ff7f0e',
                            edgecolor='black')

            ax.set_title(f"Funzione {func_id} - Response Time MEDIO (Chunk {chunk + 1}/{num_chunks})", fontsize=14,
                         fontweight='bold')
            ax.set_ylabel("Response Time (secondi)", fontsize=12)
            ax.set_xlabel("Row / Carico (req/s)", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.legend(fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"func_{func_id}_rt_mean_chunk_{chunk + 1}.png"), dpi=300)
            plt.close()

            # ==========================================
            # PLOT 2: SIM MEAN vs REAL MEDIAN
            # ==========================================
            fig, ax = plt.subplots(figsize=(14, 6))

            rects1 = ax.bar(x - width / 2, sim_rt_chunk, width, label='Simulatore (Media)', color='#1f77b4',
                            edgecolor='black')
            rects2 = ax.bar(x + width / 2, real_median_chunk, width, label='Serverledge (Mediana)', color='#2ca02c',
                            edgecolor='black')

            ax.set_title(f"Funzione {func_id} - Sim Media vs Real MEDIANA (Chunk {chunk + 1}/{num_chunks})",
                         fontsize=14, fontweight='bold')
            ax.set_ylabel("Response Time (secondi)", fontsize=12)
            ax.set_xlabel("Row / Carico (req/s)", fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=10)
            ax.legend(fontsize=12)
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"func_{func_id}_rt_median_chunk_{chunk + 1}.png"), dpi=300)
            plt.close()

    print(f"\n[SUCCESSO] Grafici di confronto RT generati in:\n -> {out_dir}")


if __name__ == "__main__":
    main()