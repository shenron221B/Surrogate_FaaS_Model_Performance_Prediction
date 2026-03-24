import sys
import os
import numpy as np


def print_usage():
    print("Uso: python3 compare_datasets.py <path_reale.npz> <path_sintetico.npz>")
    print("Esempio: python3 compare_datasets.py results/.../dataset.npz results/.../dataset_simulated_15_0_10000.npz")
    sys.exit(1)


def calc_stats(matrix):
    return np.nanmin(matrix), np.nanmax(matrix), np.nanmean(matrix)


def compare_metric(name, real_mat, syn_mat, is_percentage=False):
    _, _, r_mean = calc_stats(real_mat)
    _, _, s_mean = calc_stats(syn_mat)

    diff_abs = s_mean - r_mean

    # se la metrica reale è zero, calcoliamo solo la diff assoluta
    if r_mean == 0:
        diff_perc_str = "N/A"
    else:
        diff_perc = (diff_abs / r_mean) * 100
        diff_perc_str = f"{diff_perc:+.2f}%"

    if is_percentage:
        return f"{name:<15} | Reale: {r_mean * 100:6.2f}% | Simulatore: {s_mean * 100:6.2f}% | Gap Assoluto: {diff_abs * 100:+6.2f}%"
    else:
        return f"{name:<15} | Reale: {r_mean:6.4f}  | Simulatore: {s_mean:6.4f}  | Gap: {diff_abs:+6.4f} ({diff_perc_str})"


def main():
    if len(sys.argv) < 3:
        print_usage()

    real_path = os.path.abspath(sys.argv[1])
    syn_path = os.path.abspath(sys.argv[2])

    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print("[ERRORE] Uno dei file NPZ non esiste.")
        sys.exit(1)

    # carica dataset
    print(f"[INFO] Caricamento Dataset Reale: {os.path.basename(real_path)}")
    real_data = np.load(real_path)

    print(f"[INFO] Caricamento Dataset Sintetico: {os.path.basename(syn_path)}")
    syn_data = np.load(syn_path)

    # creazione nome file di output
    base_dir = os.path.dirname(syn_path)
    syn_name = os.path.splitext(os.path.basename(syn_path))[0]
    out_file = os.path.join(base_dir, f"{syn_name}_vs_real_statistics.txt")

    num_funcs = real_data['X'].shape[1]

    report = []
    report.append("=====================================================================")
    report.append(f" ANALISI DISCREPANZE: SISTEMA REALE vs SIMULATORE GO")
    report.append(f" Funzioni tracciate: {num_funcs}")
    report.append("=====================================================================\n")

    report.append("--- INFORMAZIONI GENERALI ---")
    report.append(
        f"REALE      -> Scenari: {real_data['X'].shape[0]} | Carico Medio: {np.nanmean(real_data['X']):.3f} req/s")
    report.append(
        f"SIMULATORE -> Scenari: {syn_data['X'].shape[0]} | Carico Medio: {np.nanmean(syn_data['X']):.3f} req/s")
    report.append("\n")

    report.append("--- CONFRONTO MEDIE GLOBALI (Sim-to-Real Gap) ---")

    # metriche temporali
    if 'RT' in real_data and 'RT' in syn_data:
        report.append(compare_metric("Response Time", real_data['RT'], syn_data['RT']))

    if 'Queue' in real_data and 'Queue' in syn_data:
        report.append(compare_metric("Queueing Time", real_data['Queue'], syn_data['Queue']))

    report.append("")
    # metriche percentuali/tassi
    if 'U' in real_data and 'U' in syn_data:
        report.append(compare_metric("Utility (U)", real_data['U'], syn_data['U'], is_percentage=True))

    if 'Success' in real_data and 'Success' in syn_data:
        report.append(compare_metric("Success Rate", real_data['Success'], syn_data['Success'], is_percentage=True))

    if 'Cold' in real_data and 'Cold' in syn_data:
        report.append(compare_metric("Cold Start Rate", real_data['Cold'], syn_data['Cold'], is_percentage=True))

    report.append("\n=====================================================================")
    report.append(" DETTAGLIO PER SINGOLA FUNZIONE (Response Time & Utility)")
    report.append("=====================================================================")

    for f in range(num_funcs):
        report.append(f"\n[ FUNZIONE {f + 1} ]")
        if 'RT' in real_data and 'RT' in syn_data:
            r_rt = real_data['RT'][:, f]
            s_rt = syn_data['RT'][:, f]
            report.append("  " + compare_metric("Response Time", r_rt, s_rt))

        if 'U' in real_data and 'U' in syn_data:
            r_u = real_data['U'][:, f]
            s_u = syn_data['U'][:, f]
            report.append("  " + compare_metric("Utility", r_u, s_u, is_percentage=True))

    # scrittura su file
    with open(out_file, "w") as f:
        f.write("\n".join(report))

    print(f"\n[SUCCESSO] Report statistico generato e salvato in:")
    print(f" -> {out_file}")

    # anteprima rapida a schermo
    print("\n--- ANTEPRIMA RAPIDA ---")
    for line in report[9:16]:
        print(line)


if __name__ == "__main__":
    main()