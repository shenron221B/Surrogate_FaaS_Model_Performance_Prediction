import sys
import os
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 npz_to_csv.py <path_al_file_dataset.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]
    if not os.path.exists(npz_path):
        print(f"[ERRORE] File non trovato: {npz_path}")
        sys.exit(1)

    # creazione del nome del file di output
    csv_path = npz_path.replace('.npz', '.csv')

    print(f"Caricamento del dataset: {npz_path}...")
    try:
        data = np.load(npz_path)
    except Exception as e:
        print(f"[ERRORE] Impossibile leggere il file .npz: {e}")
        sys.exit(1)

    keys = data.files
    print(f"Metriche trovate: {keys}")

    if 'X' not in keys:
        print("[ERRORE] Il dataset non contiene la matrice 'X' dei carichi.")
        sys.exit(1)

    num_rows, num_funcs = data['X'].shape

    # dizionario
    df_data = {}
    df_data['Row'] = np.arange(1, num_rows + 1)

    # ordine logico delle colonne
    preferred_order = ['X', 'U', 'RT', 'Success', 'Warm', 'Cold']

    # riordino delle chiavi
    ordered_keys = [k for k in preferred_order if k in keys] + [k for k in keys if k not in preferred_order]

    # srotolamento matrici in colonne
    for key in ordered_keys:
        matrix = data[key]
        for f in range(num_funcs):
            col_name = f"{key}_F{f + 1}"
            df_data[col_name] = matrix[:, f]

    # crea il DataFrame Pandas
    df = pd.DataFrame(df_data)

    df = df.round(4)

    # salva in CSV
    df.to_csv(csv_path, index=False)
    print(f"[OK] Convertito con successo! File salvato in:\n -> {csv_path}")


if __name__ == "__main__":
    main()