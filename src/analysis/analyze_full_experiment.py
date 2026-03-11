import sys
import os
import glob
import re
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.stats import gamma

plt.switch_backend('Agg')


def print_usage():
    print("Uso: python3 analyze_full_experiment.py <path_cartella_principale>")
    print("Esempio: python3 analyze_full_experiment.py serverledge/results/1f_2GBpm/del/x13")
    sys.exit(1)


def load_matrix(matrix_path):
    loads = []
    try:
        with open(matrix_path, 'r') as f:
            for line in f:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if nums:
                    loads.append([float(n) for n in nums])
        return np.array(loads)
    except Exception as e:
        print(f"[ERRORE] {e}")
        return None


# analizza i file e ritorna la durations per l'istogramma
def analyze_serverledge_txts(row_path):
    txt_files = glob.glob(os.path.join(row_path, "http_responses_func*.txt"))

    total_valid = 0
    cold_starts = 0
    warm_starts = 0
    queue_times = []
    durations = []

    for txt_path in txt_files:
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        total_valid += 1

                        if data.get('IsWarmStart') is False:
                            cold_starts += 1
                        else:
                            warm_starts += 1

                        queue_times.append(data.get('QueueingTime', 0))
                        durations.append(data.get('Duration', 0))
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            pass

    sl_metrics = {
        'total_valid': total_valid,
        'cold_starts': cold_starts,
        'warm_starts': warm_starts,
        'q_mean': np.mean(queue_times) if queue_times else 0,
        'q_max': np.max(queue_times) if queue_times else 0,
        'dur_min': np.min(durations) if durations else 0,
        'dur_mean': np.mean(durations) if durations else 0,
        'dur_median': np.median(durations) if durations else 0,
        'dur_max': np.max(durations) if durations else 0,
        'raw_durations': durations
    }
    return sl_metrics


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    if not os.path.exists(base_dir):
        print(f"Errore: La cartella {base_dir} non esiste.")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f" ANALISI COMPLETA (JMETER + SERVERLEDGE) -> {os.path.basename(base_dir)}")
    print(f"=======================================================\n")

    matrix_path = os.path.join(base_dir, "X_matrix_used.txt")
    X_matrix = load_matrix(matrix_path)

    output_dir = os.path.join(base_dir, "JTL_Plots")
    os.makedirs(output_dir, exist_ok=True)

    report_lines = []
    report_lines.append(f"REPORT DETTAGLIATO SATURAZIONE E STATISTICHE - DATASET: {os.path.basename(base_dir)}")
    report_lines.append("=" * 85)

    row_dirs = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            match = re.match(r"^row_(\d+)$", entry)
            if match:
                row_dirs.append((int(match.group(1)), full_path))

    row_dirs.sort(key=lambda x: x[0])

    for idx, row_path in row_dirs:
        matrix_idx = idx - 1
        if X_matrix is not None and 0 <= matrix_idx < len(X_matrix):
            load_vector = X_matrix[matrix_idx]
        else:
            load_vector = [-1, -1, -1]

        load_str = f"[{', '.join([str(x) for x in load_vector])}]"

        # analisi Serverledge (.txt)
        sl = analyze_serverledge_txts(row_path)
        sl_tot = sl['total_valid']
        sl_cold = sl['cold_starts']
        sl_warm = sl['warm_starts']
        cold_perc = (sl_cold / sl_tot * 100) if sl_tot > 0 else 0

        # analisi JMeter (.jtl)
        jtl_files = glob.glob(os.path.join(row_path, "*.jtl"))
        jm_tot = 0;
        jm_fail_req = 0;
        jm_fail_perc = 0
        jm_mean_rt = 0;
        jm_max_rt = 0;
        jm_min_rt = 0;
        jm_median_rt = 0
        jm_utility = 0.0

        if jtl_files:
            try:
                df = pd.read_csv(jtl_files[0])
                if not df.empty:
                    df['rt_sec'] = df['elapsed'] / 1000.0
                    jm_tot = len(df)
                    jm_fail_req = len(df[df['success'] == False])
                    jm_fail_perc = (jm_fail_req / jm_tot) * 100 if jm_tot > 0 else 0
                    jm_min_rt = df['rt_sec'].min()
                    jm_mean_rt = df['rt_sec'].mean()
                    jm_median_rt = df['rt_sec'].median()
                    jm_max_rt = df['rt_sec'].max()

                    # calcolo utility (deadline 2.5*serv_time)
                    if sl['dur_mean'] > 0 and jm_tot > 0:
                        success_df = df[df['success'] == True]
                        valid_reqs = success_df[success_df['rt_sec'] < (sl['dur_mean'] * 2.5)]
                        jm_utility = len(valid_reqs) / jm_tot
            except:
                pass

        sl_dropped = jm_tot - sl_tot

        print(
            f"Row {idx:02d} | Carico: {load_str:<15} | JMeter: {jm_tot} req ({jm_fail_perc:5.1f}% fail) | Valide: {sl_tot} | U: {jm_utility:.3f} | Cold: {sl_cold} ({cold_perc:4.1f}%)")

        # formattazione .txt esteso
        report_lines.append(f"\nROW {idx:02d} | CARICO: {load_str}")
        report_lines.append(f"  [JMeter Client]")
        report_lines.append(f"    - Richieste Inviate : {jm_tot}")
        report_lines.append(f"    - Fallite / Timeout : {jm_fail_req} ({jm_fail_perc:.1f}%)")
        report_lines.append(f"    - Utilità (U)       : {jm_utility:.4f} (RT < 2.5 * Avg_Service_Time)")
        report_lines.append(f"    - RT Minimo         : {jm_min_rt:.3f}s")
        report_lines.append(f"    - RT Medio          : {jm_mean_rt:.3f}s")
        report_lines.append(f"    - RT Mediano        : {jm_median_rt:.3f}s")
        report_lines.append(f"    - RT Massimo        : {jm_max_rt:.3f}s")
        report_lines.append(f"  [Server Interno (Serverledge)]")
        report_lines.append(f"    - Esecuzioni Valide : {sl_tot}")
        report_lines.append(f"    - Scartate/Respinte : {sl_dropped} (API Gateway Drop o Timeout)")
        report_lines.append(f"    - Warm Starts       : {sl_warm}")
        report_lines.append(f"    - Cold Starts       : {sl_cold} ({cold_perc:.1f}% delle valide)")
        report_lines.append(f"    - Tempo in Coda     : Medio = {sl['q_mean']:.4f}s | Max = {sl['q_max']:.4f}s")
        report_lines.append(
            f"    - Tempo di Calcolo  : Min = {sl['dur_min']:.4f}s | Mediano = {sl['dur_median']:.4f}s | Medio = {sl['dur_mean']:.4f}s | Max = {sl['dur_max']:.4f}s")

        safe_load_str = "_".join([str(x).replace('.', 'p') for x in load_vector])

        # scatter plot JMeter
        if jtl_files and 'df' in locals() and not df.empty:
            plt.figure(figsize=(8, 4))
            success_df = df[df['success'] == True]
            fail_df = df[df['success'] == False]
            plt.scatter((success_df['timeStamp'] - df['timeStamp'].min()) / 1000, success_df['rt_sec'], alpha=0.5, s=10,
                        label='OK', color='blue')
            if not fail_df.empty:
                plt.scatter((fail_df['timeStamp'] - df['timeStamp'].min()) / 1000, fail_df['rt_sec'], alpha=0.8, s=20,
                            label='Fallite', color='red', marker='x')
            plt.title(f"Latenze - Row {idx:02d} - Carico {load_str}")
            plt.xlabel("Tempo trascorso del test (Secondi)")
            plt.ylabel("Tempo di Risposta (Secondi)")
            plt.axhline(y=10.0, color='r', linestyle='--', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"scatter_row_{idx:02d}_load_{safe_load_str}.png"), dpi=150)
            plt.close()

        # istogramma tempi di servizio
        durations = sl['raw_durations']
        if len(durations) > 1:
            plt.figure(figsize=(8, 4))
            # rimozione eventuali outlier estremi (fino al 99° percentile)
            p99 = np.percentile(durations, 99)
            filtered_durations = [d for d in durations if d <= p99]

            plt.hist(filtered_durations, bins=30, density=True, alpha=0.6, color='g', label='Dati Reali (Duration)')

            # fitting con curva gamma
            try:
                shape, loc, scale = gamma.fit(filtered_durations, floc=0)
                x_val = np.linspace(0, max(filtered_durations), 100)
                y_val = gamma.pdf(x_val, shape, loc, scale)
                plt.plot(x_val, y_val, 'r-', lw=2, label=f'Fit Gamma\n(shape={shape:.2f})')
            except:
                pass

            plt.title(f"Distribuzione Tempi di Servizio - Row {idx:02d}")
            plt.xlabel("Tempo (Secondi)")
            plt.ylabel("Frequenza")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"histogram_row_{idx:02d}_load_{safe_load_str}.png"), dpi=150)
            plt.close()

    # salvataggio report
    report_path = os.path.join(base_dir, "detailed_analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n[SUCCESSO] Report esteso salvato in: {report_path}")
    print(f"[SUCCESSO] Grafici e Istogrammi salvati in: {output_dir}")


if __name__ == "__main__":
    main()