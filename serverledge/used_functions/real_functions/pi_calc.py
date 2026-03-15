import random
import time


def invoke(args):
    # prende il numero di iterazioni dal JSON (default 10000)
    iterations = args.get("iterations", 10000)
    inside = 0
    start = time.time()

    for _ in range(iterations):
        x, y = random.random(), random.random()
        if x * x + y * y <= 1.0:
            inside += 1

    pi = (inside / iterations) * 4
    duration = time.time() - start

    return {"pi": pi, "duration_sec": duration, "iterations": iterations}