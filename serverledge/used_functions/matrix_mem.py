import time


def invoke(args, context):
    size = int(args.get("size", 1000))

    start = time.time()
    matrix = [[i * j for j in range(size)] for i in range(size)]
    sum_val = sum(matrix[i][i] for i in range(size))
    duration = time.time() - start

    return {"diagonal_sum": sum_val, "duration_sec": duration, "size": size}

