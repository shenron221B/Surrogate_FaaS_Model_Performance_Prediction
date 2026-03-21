import sys
import os
import json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gamma

FUNC_OFFSET = 1
SYSTEM_MEMORY = 3072
MEM_DEMAND = 256

def load_matrix(matrix_file, n_cols):
    if os.path.exists(matrix_file):
        return np.loadtxt(matrix_file, ndmin=2)
    print(f"[WARN] Impossibile trovare {matrix_file}")
    return np.zeros((1, n_cols))


def compute_utility_mean(df, func_id, mean_dur):
    label = f"Invoke_func_{func_id}"
    df_func = df[df["label"] == label]
    total_invocation = len(df_func)

    if total_invocation > 0:
        sl_dur_mean = mean_dur if mean_dur > 0 else 0.200
        utility_threshold = max(sl_dur_mean * 2.5, 0.500)

        valid = df_func[(df_func['success'] == True) & ((df_func['elapsed'] / 1000.0) < utility_threshold)]
        utility = len(valid) / total_invocation
        return utility
    return 0.0


def compute_response_time_median(df, func_id):
    df_func = df[df["label"] == f"Invoke_func_{func_id}"]
    df_func = df_func[df_func["success"] == True]
    return np.median(df_func["elapsed"] / 1000.0) if not df_func.empty else 0.0


def read_http_file_extended(filename):
    durations, inits, queue_times = [], [], []
    cold_count, total_count = 0, 0
    if not os.path.exists(filename): return [], 0, 0, 0, 0, 0
    with open(filename, "r") as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)

                if "Success" in data and not data["Success"]: continue

                total_count += 1
                q_t = data.get("QueueingTime", 0)
                queue_times.append(q_t)
                init_time = data.get("InitTime", 0)
                duration = data.get("Duration", 0)
                is_warm = data.get("IsWarmStart", False)
                if is_warm == False:
                    cold_count += 1
                    inits.append(max(0, init_time - q_t))
                    durations.append(duration)
                else:
                    durations.append(duration + max(0, init_time - q_t))
            except json.JSONDecodeError:
                continue
    mean_dur = np.mean(durations) if durations else 0
    mean_init = np.nanmean(inits) if inits else 0
    mean_queue = np.mean(queue_times) if queue_times else 0
    return durations, mean_dur, mean_init, mean_queue, cold_count, total_count


def calculate_net_overhead(filename, df, func_id):
    diff = []
    df_func = df[df["label"] == f"Invoke_func_{func_id}"]
    df_func = df_func[df_func["success"] == True]
    jmeter_latencies = df_func["elapsed"].values
    if not os.path.exists(filename): return []
    count = 0
    with open(filename, "r") as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                if "Success" in data and not data["Success"]: continue

                serv_rt = data.get("ResponseTime", 0)
                if count < len(jmeter_latencies):
                    overhead = (jmeter_latencies[count] / 1000.0) - serv_rt
                    if overhead > 0: diff.append(overhead)
                    count += 1
            except (json.JSONDecodeError, IndexError):
                continue
    return diff


def main():
    if len(sys.argv) < 3:
        print("Errore: Passare la cartella dei risultati e il numero di funzioni.")
        print("Uso: python3 create_dataset.py <Dir> <Num_Funcs>")
        sys.exit(1)

    RESULTS_DIR = sys.argv[1]
    FUNCTIONS = int(sys.argv[2])

    CONFIG_OUT = os.path.join(RESULTS_DIR, "simulator-conf.yml")
    SERV_TRAIN = os.path.join(RESULTS_DIR, "dataset.npz")
    MATRIX_FILE = os.path.join(RESULTS_DIR, "X_matrix_used.txt")

    if not os.path.exists(RESULTS_DIR):
        print(f"Errore: La directory {RESULTS_DIR} non esiste.")
        sys.exit(1)

    X = load_matrix(MATRIX_FILE, n_cols=FUNCTIONS)
    rows = sorted([d for d in os.listdir(RESULTS_DIR) if d.startswith("row")], key=lambda x: int(x.split('_')[1]))

    mean_inits, mean_durations, all_durations, cvs, net_overheads = [], [], [], [], []
    for row in rows:
        row_path = os.path.join(RESULTS_DIR, row)
        row_durations_list, row_inits_list = [], []

        jtl_files = [f for f in os.listdir(row_path) if f.endswith(".jtl")]
        df = pd.read_csv(os.path.join(row_path, jtl_files[0])) if jtl_files else None

        for func_id in range(FUNC_OFFSET, FUNC_OFFSET + FUNCTIONS):
            http_file = os.path.join(row_path, f"http_responses_func{func_id}.txt")
            durations, dur_mean, init_mean, _, _, _ = read_http_file_extended(http_file)
            row_durations_list.append(durations)
            row_inits_list.append(init_mean)

            if df is not None:
                overhead = calculate_net_overhead(http_file, df, func_id)
                net_overheads.extend(overhead)

        mean_durations.append([np.mean(d) if len(d) > 0 else 0 for d in row_durations_list])
        mean_inits.append(row_inits_list)
        all_durations.append(row_durations_list)

    mean_net_overhead = float(np.median(net_overheads)) if net_overheads else 0
    flat_durations = []
    for i in range(FUNCTIONS):
        durations_func = []
        for r in range(len(rows)):
            if len(all_durations[r]) > i: durations_func.extend(all_durations[r][i])
        flat_durations.append(durations_func)

    for dur_list in flat_durations:
        if len(dur_list) > 10:
            shape, loc, scale = gamma.fit(dur_list, floc=0)
            cvs.append(float(1 / np.sqrt(shape)))
        else:
            cvs.append(1.0)

    config = {
        "data": "data/serverledge",
        "seeds": [1], "qlen": 30, "functions": FUNCTIONS, "system_memory": SYSTEM_MEMORY, "poisson_arrivals": True,
        "deadline_coeff": 2.5, "queue_policy": "fifo",
        "net_overhead": mean_net_overhead, "serv_time_duration": np.mean(np.array(mean_durations), axis=0).tolist(),
        "serv_time_cvs": cvs,
        "init_times": np.nanmean(np.array(mean_inits), axis=0).tolist(), "mem_demands": [MEM_DEMAND] * FUNCTIONS,
        "arv_rates": [1] * FUNCTIONS, "parallelism": 1,
    }
    with open(CONFIG_OUT, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"\n[OK] Configurazione simulatore salvata.")

    print("\n--- ANALISI DATASET CURATO ---")
    RT_matrix, U_matrix = [], []

    for idx_row, row in enumerate(rows):
        if idx_row >= len(X): break
        row_path = os.path.join(RESULTS_DIR, row)
        RT_row, U_row = [], []
        jtl_files = [f for f in os.listdir(row_path) if f.endswith(".jtl")]
        df = pd.read_csv(os.path.join(row_path, jtl_files[0])) if jtl_files else pd.DataFrame()

        for f in range(FUNC_OFFSET, FUNC_OFFSET + FUNCTIONS):
            file_txt = os.path.join(row_path, f"http_responses_func{f}.txt")
            _, mean_dur, _, _, _, _ = read_http_file_extended(file_txt)

            if not df.empty:
                rt_client = compute_response_time_median(df, f)
                utility = compute_utility_mean(df, f, mean_dur)
            else:
                rt_client = 0
                utility = 0

            RT_row.append(rt_client)
            U_row.append(utility)

        RT_matrix.append(RT_row)
        U_matrix.append(U_row)

    np.savez(SERV_TRAIN, RT=np.array(RT_matrix), U=np.array(U_matrix), X=X)
    print(f"[OK] Dataset salvato in: {SERV_TRAIN}")


if __name__ == "__main__":
    main()