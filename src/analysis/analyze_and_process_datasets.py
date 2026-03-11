import os
import sys
import json
import numpy as np
import pandas as pd

def parse_real_logs(target_dir, num_funcs):
    total_reqs = 0
    cold_starts = 0
    queue_times = []

    for row_dir in os.listdir(target_dir):
        full_row_path = os.path.join(target_dir, row_dir)
        if not os.path.isdir(full_row_path) or not row_dir.startswith("row_"):
            continue

        for f_idx in range(1, num_funcs + 1):
            log_file = os.path.join(full_row_path, f"http_responses_func{f_idx}.txt")
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            total_reqs += 1
                            if data.get("IsWarmStart") is False:
                                cold_starts += 1
                            queue_times.append(data.get("QueueingTime", 0))
                        except json.JSONDecodeError:
                            continue

    return total_reqs, cold_starts, queue_times

def export_to_csv(X, RT, U, out_path, is_synthetic=False, Cold=None):
    rows = []
    prefix = "row_sim" if is_synthetic else "row"
    num_funcs = X.shape[1]

    for i in range(len(X)):
        row = [f"{prefix}_{i + 1}"]
        row.extend(X[i])
        row.extend(RT[i])
        row.extend(U[i])
        if is_synthetic and Cold is not None:
            row.extend(Cold[i])
        rows.append(row)

    cols = ["Row"]
    cols.extend([f"Load_F{j+1}" for j in range(num_funcs)])
    cols.extend([f"RT_F{j+1}" for j in range(num_funcs)])
    cols.extend([f"Util_F{j+1}" for j in range(num_funcs)])

    if is_synthetic and Cold is not None:
        cols.extend([f"Cold_F{j+1}" for j in range(num_funcs)])

    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_path, index=False)

def analyze_dataset(target_dir):
    if not os.path.exists(target_dir):
        print(f"[ERRORE] La directory {target_dir} non esiste.")
        sys.exit(1)

    print(f"\n[INFO] Avvio pipeline di analisi su: {target_dir}")
    report_path = os.path.join(target_dir, "dataset_statistics.txt")

    if not os.path.exists(report_path):
        with open(report_path, "w") as f:
            f.write("=========================================================\n")
            f.write(f" REPORT STATISTICO DATASET: {os.path.basename(target_dir)}\n")
            f.write("=========================================================\n\n")

    npz_files = [f for f in os.listdir(target_dir) if f.endswith(".npz")]
    new_lines_to_append = []

    for npz_file in npz_files:
        base_name = os.path.splitext(npz_file)[0]
        csv_file = f"{base_name}.csv"
        csv_path = os.path.join(target_dir, csv_file)
        npz_path = os.path.join(target_dir, npz_file)

        if os.path.exists(csv_path):
            print(f"[SKIP] File '{npz_file}' già analizzato.")
            continue

        print(f"[PROCESS] Analizzo nuovo dataset: {npz_file}")

        data = np.load(npz_path)
        X, RT, U = data['X'], data['RT'], data['U']
        num_funcs = X.shape[1]

        is_real = (npz_file == "dataset.npz" or npz_file.startswith("data_"))
        file_report = []

        if is_real:
            tot_req, c_starts, q_times = parse_real_logs(target_dir, num_funcs)
            c_perc = (c_starts / tot_req * 100) if tot_req > 0 else 0

            zeros_rt = np.sum(RT == 0)
            zeros_u = np.sum(U == 0)
            zero_rows = [f"Row {i + 1} F{f + 1}" for i in range(len(RT)) for f in range(num_funcs) if RT[i][f] == 0]

            file_report.append("--- SISTEMA REALE (Serverledge) ---")
            file_report.append(f"File sorgente: {npz_file}")
            file_report.append(f"Dimensioni (Scenari x Funzioni): X {X.shape}, RT {RT.shape}, U {U.shape}")
            file_report.append(f"Anomalie: RT nulli: {zeros_rt}, Utility nulle: {zeros_u}, Valori NaN: {np.sum(np.isnan(RT))}")
            if zero_rows:
                file_report.append(f"          Dettaglio RT=0: {', '.join(zero_rows)}")
            file_report.append(f"Carico req/s: Min={np.min(X):.3f} | Max={np.max(X):.3f} | Medio={np.mean(X):.3f}")
            file_report.append(f"Resp. Time:   Min={np.min(RT):.4f}s | Max={np.max(RT):.4f}s | Medio={np.mean(RT):.4f}s")
            file_report.append(f"Utility:      Media={np.mean(U):.4f}")
            file_report.append(f"Cold Starts:  {c_starts} su {tot_req} richieste ({c_perc:.2f}%)")
            if q_times:
                file_report.append(f"Queue Time:   Min={np.min(q_times):.6f}s | Max={np.max(q_times):.6f}s | Medio={np.mean(q_times):.6f}s")

            export_to_csv(X, RT, U, csv_path, is_synthetic=False)

        else:
            Cold_s = data['Cold'] if 'Cold' in data else None
            avg_U = np.mean(U, axis=1)

            file_report.append("--- SIMULATORE SINTETICO (Go) ---")
            file_report.append(f"File sorgente: {npz_file}")
            file_report.append(f"Dimensioni: X {X.shape}, RT {RT.shape}, U {U.shape}")
            file_report.append(f"Anomalie: RT nulli: {np.sum(RT == 0)}, Valori NaN: {np.sum(np.isnan(RT))}")
            file_report.append(f"Carico req/s: Min={np.min(X):.3f} | Max={np.max(X):.3f} | Medio={np.mean(X):.3f}")
            file_report.append(f"Resp. Time:   Min={np.min(RT):.4f}s | Max={np.max(RT):.4f}s | Medio={np.mean(RT):.4f}s")
            file_report.append(f"Utility:      Media={np.mean(U):.4f}")
            file_report.append(f"Distribuzione Utility per Scenario:")
            file_report.append(f"  - Scenari con U > 0.90: {np.sum(avg_U > 0.90)}")
            file_report.append(f"  - Scenari con U > 0.50: {np.sum(avg_U > 0.50)}")
            file_report.append(f"  - Scenari con U < 0.10: {np.sum(avg_U < 0.10)}")
            if Cold_s is not None:
                file_report.append(f"Cold Start:   Media Stimata={np.mean(Cold_s) * 100:.2f}%")
            file_report.append("Queue Time:   Non calcolato separatamente nel sintetico.")

            export_to_csv(X, RT, U, csv_path, is_synthetic=True, Cold=Cold_s)

        file_report.append("\n")
        new_lines_to_append.extend(file_report)
        print(f"  -> CSV generato: {csv_file}")

    if new_lines_to_append:
        with open(report_path, "a") as f:
            f.write("\n".join(new_lines_to_append))
        print(f"\n[SUCCESSO] Statistiche aggiornate in: {report_path}")
    else:
        print(f"\n[INFO] Nessun nuovo file NPZ da analizzare. Report testuale invariato.")

    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_and_process_datasets.py <percorso_cartella_risultati>")
        sys.exit(1)

    target_directory = sys.argv[1]
    analyze_dataset(target_directory)