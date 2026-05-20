import sys
import numpy as np


def print_usage():
    print("Uso: python3 interpolate_dataset.py <input.npz> <output.npz> <idx_start> <idx_end> <num_nuove_righe>")
    print("Esempio: python3 interpolate_dataset.py dataset.npz dataset_smooth.npz 4 5 3")
    print("Questo inserirà 3 nuove righe interpolate gradualmente tra l'indice 4 e l'indice 5.")
    sys.exit(1)


def main():
    if len(sys.argv) < 6:
        print_usage()

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    try:
        idx_start = int(sys.argv[3])
        idx_end = int(sys.argv[4])
        num_new_rows = int(sys.argv[5])
    except ValueError:
        print("[ERRORE] Indici e numero di righe devono essere interi.")
        sys.exit(1)

    if idx_end <= idx_start:
        print("[ERRORE] idx_end deve essere maggiore di idx_start.")
        sys.exit(1)

    data = np.load(input_file)
    new_data = {}

    print(f"\n[INFO] Caricamento {input_file}")
    print(f"[INFO] Interpolazione di {num_new_rows} righe tra indice {idx_start} e {idx_end}...")

    for key in data.files:
        mat = data[key]

        # Estrai le due righe "ancora"
        row_start = mat[idx_start]
        row_end = mat[idx_end]

        # Genera i valori interpolati linearmente.
        # num_new_rows + 2 serve per includere gli estremi, che poi scartiamo con [1:-1]
        interpolated_vals = np.linspace(row_start, row_end, num=num_new_rows + 2)[1:-1]

        # Inserisci i nuovi valori subito dopo idx_start
        new_mat = np.insert(mat, idx_start + 1, interpolated_vals, axis=0)
        new_data[key] = new_mat

    # Salvataggio
    np.savez(output_file, **new_data)

    original_len = data['X'].shape[0]
    new_len = new_data['X'].shape[0]
    print(f"[OK] Salvato in {output_file}")
    print(f" -> Righe originali: {original_len}")
    print(f" -> Righe finali:    {new_len}")


if __name__ == "__main__":
    main()