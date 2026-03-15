import hashlib
import os
import time


def invoke(args):
    password = args.get("password", "test_password_for_benchmark").encode('utf-8')
    # iterazioni dell'algoritmo
    iterations = args.get("iterations", 300000)
    salt = os.urandom(16)

    start = time.time()
    # generazione dell'hash
    key = hashlib.pbkdf2_hmac('sha256', password, salt, iterations)
    duration = time.time() - start

    return {"hash": key.hex(), "duration_sec": duration, "iterations": iterations}