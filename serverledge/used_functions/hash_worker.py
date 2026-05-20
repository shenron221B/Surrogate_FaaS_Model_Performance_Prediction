import hashlib
import os
import time


def invoke(args, context):
    password = str(args.get("password", "test_password_for_benchmark")).encode('utf-8')
    iterations = int(args.get("iterations", 300000))
    salt = os.urandom(16)

    start = time.time()
    key = hashlib.pbkdf2_hmac('sha256', password, salt, iterations)
    duration = time.time() - start

    return {"hash": key.hex(), "duration_sec": duration, "iterations": iterations}


