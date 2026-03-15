import time


def invoke(args):
    # dimensione della matrice (default 1000x1000)
    size = args.get("size", 1000)

    start = time.time()
    # allocazione grossa matrice in RAM
    matrix = [[i * j for j in range(size)] for i in range(size)]

    # calcola la somma della diagonale
    sum_val = sum(matrix[i][i] for i in range(size))
    duration = time.time() - start

    return {"diagonal_sum": sum_val, "duration_sec": duration, "size": size}