import sys
import os
import glob
import re
import pandas as pd
import numpy as np
import json


def print_usage():
    print("Uso: python3 analyze_experiment_single_func.py <path_cartella_principale>")
    print("Esempio: python3 analyze_experiment_single_func.py serverledge/results/3f_3GBpm/q_len_5/256/x17")
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


def analyze_serverledge_txt(txt_path):
    total_valid, cold_starts, warm_starts = 0, 0, 0
    queue_times, durations = [], []

    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith("{"):
                    try:
                        data = json.loads(line)
                        if data.get('Success') is True:
                            total_valid += 1
                            if data.get('IsWarmStart') is False:
                                cold_starts += 1
                            else:
                                warm_starts += 1
                            queue_times.append(data.get('QueueingTime', 0))
                            durations.append(data.get('Duration', 0))
                    except:
                        pass

    return {
        'total_valid': total_valid,
        'cold_starts': cold_starts,
        'warm_starts': warm_starts,
        'q_mean': np.mean(queue_times) if queue_times else 0,
        'q_max': np.max(queue_times) if queue_times else 0,
        'dur_min': np.min(durations) if durations else 0,
        'dur_mean': np.mean(durations) if durations else 0,
        'dur_median': np.median(durations) if durations else 0,
        'dur_max': np.max(durations) if durations else 0
    }


def main():
    if len(sys.argv) < 2:
        print_usage()

    base_dir = os.path.abspath(sys.argv[1])
    if not os.path.exists(base_dir):
        print(f"Errore: La cartella {base_dir} non esiste.")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f" ANALISI MULTI-FUNZIONE UNIVERSALE -> {os.path.basename(base_dir)}")
    print(f"{'=' * 80}\n")

    matrix_path = os.path.join(base_dir, "X_matrix_used.txt")
    X_matrix = load_matrix(matrix_path)

    report_lines = []
    report_lines.append(f"REPORT DETTAGLIATO SEPARATO PER FUNZIONE - DATASET: {os.path.basename(base_dir)}")
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
        load_vector = X_matrix[matrix_idx] if (X_matrix is not None and 0 <= matrix_idx < len(X_matrix)) else []
        load_str = f"[{', '.join([str(x) for x in load_vector])}]"

        report_lines.append(f"\nROW {idx:02d} | CARICO COMPLESSIVO: {load_str}")
        report_lines.append("-" * 60)

        jtl_files = glob.glob(os.path.join(row_path, "*.jtl"))
        df_jmeter = pd.DataFrame()
        if jtl_files:
            try:
                df_jmeter = pd.read_csv(jtl_files[0])
                df_jmeter['rt_sec'] = df_jmeter['elapsed'] / 1000.0
            except:
                pass

        # trova dinamicamente tutte le funzioni testate in questa riga
        func_files = glob.glob(os.path.join(row_path, "http_responses_func*.txt"))
        func_ids = []
        for f in func_files:
            match = re.search(r"http_responses_func(\d+)\.txt", f)
            if match: func_ids.append(int(match.group(1)))
        func_ids.sort()

        for f_id in func_ids:
            sl_path = os.path.join(row_path, f"http_responses_func{f_id}.txt")
            sl = analyze_serverledge_txt(sl_path)

            # filtra JMeter per la singola funzione
            jm_tot, jm_fail_req, jm_utility = 0, 0, 0.0
            jm_mean_rt, jm_min_rt, jm_max_rt, jm_median_rt = 0, 0, 0, 0

            if not df_jmeter.empty:
                df_func = df_jmeter[df_jmeter['label'].str.contains(f"func_{f_id}", case=False, na=False)]
                jm_tot = len(df_func)
                if jm_tot > 0:
                    jm_fail_req = len(df_func[df_func['success'] == False])
                    jm_min_rt = df_func['rt_sec'].min()
                    jm_mean_rt = df_func['rt_sec'].mean()
                    jm_median_rt = df_func['rt_sec'].median()
                    jm_max_rt = df_func['rt_sec'].max()

                    # utilità calcolata sul service time della specifica funzione
                    if sl['dur_mean'] > 0:
                        success_df = df_func[df_func['success'] == True]
                        valid_reqs = success_df[success_df['rt_sec'] < (sl['dur_mean'] * 2.5)]
                        jm_utility = len(valid_reqs) / jm_tot

            jm_fail_perc = (jm_fail_req / jm_tot * 100) if jm_tot > 0 else 0
            sl_dropped = max(0, jm_tot - sl['total_valid'])
            cold_perc = (sl['cold_starts'] / sl['total_valid'] * 100) if sl['total_valid'] > 0 else 0

            # carico specifico della funzione
            spec_load = load_vector[f_id - 1] if (f_id - 1) < len(load_vector) else "?"

            print(
                f"Row {idx:02d} | F{f_id} (Carico: {spec_load}) -> JMeter Drop: {jm_fail_perc:5.1f}% | Coda: {sl['q_mean']:.4f}s | U: {jm_utility:.3f}")

            report_lines.append(f"  [FUNZIONE {f_id}] (Carico Applicato: {spec_load})")
            report_lines.append(
                f"    - Inviate (JMeter)  : {jm_tot} | Fallite: {jm_fail_req} ({jm_fail_perc:.1f}%) | U: {jm_utility:.4f}")
            report_lines.append(
                f"    - Latenza (JMeter)  : Min={jm_min_rt:.3f}s | Med={jm_median_rt:.3f}s | Media={jm_mean_rt:.3f}s | Max={jm_max_rt:.3f}s")
            report_lines.append(f"    - Valide (Server)   : {sl['total_valid']} | Scartate GW: {sl_dropped}")
            report_lines.append(
                f"    - Cold Starts       : {sl['cold_starts']} ({cold_perc:.1f}%) | Warm: {sl['warm_starts']}")
            report_lines.append(f"    - Tempo Coda        : Media={sl['q_mean']:.4f}s | Max={sl['q_max']:.4f}s")
            report_lines.append(f"    - Tempo Servizio    : Media={sl['dur_mean']:.4f}s | Max={sl['dur_max']:.4f}s")
            report_lines.append(f"    " + "." * 50)

    report_path = os.path.join(base_dir, "detailed_analysis_report_single_func.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print(f"\n[SUCCESSO] Report separato salvato in: {report_path}")


if __name__ == "__main__":
    main()