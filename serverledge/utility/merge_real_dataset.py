import sys
import os
import glob
import numpy as np
import subprocess


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 merge_real_dataset.py <path_directory_principale>")
        print("Es: python3 merge_real_dataset.py /root/.../poisson_600")
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])

    # cerca dataset.npz in tutte le sottocartelle immediate
    npz_files = glob.glob(os.path.join(root_dir, "*", "dataset.npz"))

    if not npz_files:
        print(f"[ERRORE] Nessun dataset.npz trovato nelle sottocartelle di {root_dir}")
        sys.exit(1)

    print(f"\n--- MERGE DATASET NPZ ---")
    print(f"Trovati {len(npz_files)} file NPZ da unire.")

    merged_data = {}

    for f in npz_files:
        try:
            data = np.load(f)
            for key in data.files:
                if key not in merged_data:
                    merged_data[key] = []
                merged_data[key].append(data[key])
        except Exception as e:
            print(f"[ERRORE] Impossibile leggere {f}: {e}")

    final_data = {}
    total_rows = 0
    for key in merged_data:
        # concatena tutte le matrici lungo l'asse 0 (le righe)
        final_data[key] = np.concatenate(merged_data[key], axis=0)
        total_rows = final_data[key].shape[0]

    # salva il master Dataset nella cartella radice
    out_path = os.path.join(root_dir, "dataset.npz")
    np.savez(out_path, **final_data)

    print(f"[OK] Master Dataset unito ({total_rows} righe totali). Salvato in:\n -> {out_path}")

    # conversione CSV
    print("\n[INFO] Avvio conversione in CSV...")

    # trova la cartella in cui si trova questo script per richiamare npz_to_csv.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_script = os.path.join(script_dir, "npz_to_csv.py")

    if os.path.exists(csv_script):
        try:
            subprocess.run(["python3", csv_script, out_path], check=True)
            print("[SUCCESSO] Pipeline di merge completata perfettamente!")
        except subprocess.CalledProcessError as e:
            print(f"[ERRORE] Fallimento durante la conversione CSV: {e}")
    else:
        print(f"[ATTENZIONE] Script {csv_script} non trovato. Conversione in CSV saltata.")


if __name__ == "__main__":
    main()