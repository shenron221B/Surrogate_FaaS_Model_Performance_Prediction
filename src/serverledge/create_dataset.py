import sys
import os
import json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import gamma

# costanti infrastrutturali
FUNC_OFFSET = 1
SYSTEM_MEMORY = 2048
MEM_DEMAND = [256, 256]
CPU_CORES = 8
CPU_DEMAND = 1.0
QUEUE_LENGTH = 5


# caricamento della matrice dei carichi
def load_matrix(matrix_file, n_cols):
    if os.path.exists(matrix_file):
        return np.loadtxt(matrix_file, ndmin=2)
    print(f"[WARN] Impossibile trovare {matrix_file}")
    return np.zeros((1, n_cols))


# calcolo utils per QoS: U = 2.5*E[S]
def compute_utility_mean(df, func_id, mean_dur):
    label = f"Invoke_func_{func_id}"
    df_func = df[df["label"] == label]
    total_invocation = len(df_func)

    if total_invocation > 0:
        sl_dur_mean = mean_dur if mean_dur > 0 else 0.200
        utility_threshold = sl_dur_mean * 2.5
        valid = df_func[(df_func['success'] == True) & ((df_func['elapsed'] / 1000.0) < utility_threshold)]
        return len(valid) / total_invocation
    return 0.0


def compute_response_time_median(df, func_id):
    df_func = df[df["label"] == f"Invoke_func_{func_id}"]
    df_func = df_func[df_func["success"] == True]
    return np.median(df_func["elapsed"] / 1000.0) if not df_func.empty else 0.0


# lettura dai file di risposta per la creazione del dataset
def read_http_file_extended(filename):
    durations, inits, queue_times = [], [], []
    dismiss_times = []
    cold_count, warm_count = 0, 0
    if not os.path.exists(filename):
        return [], 0, 0, 0, 0, 0, 0, 0

    with open(filename, "r") as f:
        for line in f:
            try:
                line = line.strip()
                if not line: continue
                data = json.loads(line)

                if "Success" in data and not data["Success"]: continue

                d_t = data.get("DismissTime", 0.0)
                if d_t > 0: dismiss_times.append(d_t)

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
                    warm_count += 1
                    durations.append(duration + max(0, init_time - q_t))
            except json.JSONDecodeError:
                continue

    mean_dur = np.mean(durations) if durations else 0
    mean_init = np.nanmean(inits) if inits else np.nan
    mean_queue = np.mean(queue_times) if queue_times else 0
    init_75 = np.percentile(inits, 75) if inits else np.nan
    mean_dismiss = np.mean(dismiss_times) if dismiss_times else np.nan
    
    return durations, mean_dur, mean_init, mean_queue, cold_count, warm_count, init_75, mean_dismiss

def calculate_net_overhead_median(filename, df, func_id):
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

    print("\n--- AVVIO GENERAZIONE SIMULATOR CONF (YAML) ---")
    mean_inits, mean_durations, all_durations, cvs, net_overheads = [], [], [], [], []
    mean_dismisses = []
    for row in rows:
        row_path = os.path.join(RESULTS_DIR, row)
        row_durations_list, row_inits_list = [], []

        jtl_files = [f for f in os.listdir(row_path) if f.endswith(".jtl")]
        df = pd.read_csv(os.path.join(row_path, jtl_files[0])) if jtl_files else None

        for func_id in range(FUNC_OFFSET, FUNC_OFFSET + FUNCTIONS):
            http_file = os.path.join(row_path, f"http_responses_func{func_id}.txt")
            durations, dur_mean, init_mean, _, _, _, _, dismiss_mean = read_http_file_extended(http_file)
            row_durations_list.append(durations)
            row_inits_list.append(init_mean)
            mean_dismisses.append(dismiss_mean)

            if df is not None:
                overhead = calculate_net_overhead_median(http_file, df, func_id)
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

    global_dismiss = float(np.nanmean(mean_dismisses)) if not np.isnan(np.nanmean(mean_dismisses)) else 0.0

    config = {
        "data": "data/serverledge_results",
        "seeds": [1], "qlen": QUEUE_LENGTH, "functions": FUNCTIONS, "system_memory": SYSTEM_MEMORY,
        "poisson_arrivals": True,
        "deadline_coeff": 2.5, "queue_policy": "fifo",
        "net_overhead": mean_net_overhead,
        "dismiss_overhead": global_dismiss,
        "serv_time_duration": np.mean(np.array(mean_durations), axis=0).tolist(),
        "serv_time_cvs": cvs,
        "init_times": np.nanmean(np.array(mean_inits), axis=0).tolist(), "mem_demands": MEM_DEMAND,
        "arv_rates": [1] * FUNCTIONS, "parallelism": 1,
    }
    with open(CONFIG_OUT, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"[OK] Configurazione simulatore salvata: {CONFIG_OUT}")

    print("\n--- AVVIO ESTRAZIONE FEATURE ML (NPZ) ---")
    RT_matrix, U_matrix = [], []
    Warm_matrix, Cold_matrix, Success_matrix = [], [], []
    Queue_matrix, Init_matrix, Init_75_matrix, NetOv_matrix = [], [], [], []
    PoolMem_matrix, FuncMem_matrix, Cpus_matrix = [], [], []
    FuncCpu_matrix, QueueLen_matrix = [], []

    for idx_row, row in enumerate(rows):
        if idx_row >= len(X): break
        row_path = os.path.join(RESULTS_DIR, row)

        RT_row, U_row = [], []
        Warm_row, Cold_row, Success_row = [], [], []
        Queue_row, Init_row, Init_75_row, NetOv_row = [], [], [], []
        PoolMem_row, FuncMem_row, Cpus_row = [], [], []
        FuncCpu_row, QueueLen_row = [], []

        jtl_files = [f for f in os.listdir(row_path) if f.endswith(".jtl")]
        df = pd.read_csv(os.path.join(row_path, jtl_files[0])) if jtl_files else pd.DataFrame()

        for f in range(FUNC_OFFSET, FUNC_OFFSET + FUNCTIONS):
            file_txt = os.path.join(row_path, f"http_responses_func{f}.txt")

            _, mean_dur, mean_init, mean_queue, cold_count, warm_count, init_75, _ = read_http_file_extended(file_txt)

            total_valid = warm_count + cold_count
            warm_rate = warm_count / total_valid if total_valid > 0 else 0.0
            cold_rate = cold_count / total_valid if total_valid > 0 else 0.0

            if not df.empty:
                rt_client = compute_response_time_median(df, f)
                utility = compute_utility_mean(df, f, mean_dur)

                diffs_ov = calculate_net_overhead_median(file_txt, df, f)
                net_overhead = float(np.median(diffs_ov)) if diffs_ov else 0.0

                df_func = df[df["label"] == f"Invoke_func_{f}"]
                total_sent = len(df_func)
                success_count = len(df_func[df_func["success"] == True])
                success_rate = success_count / total_sent if total_sent > 0 else 0.0
            else:
                rt_client, utility, success_rate, net_overhead = 0.0, 0.0, 0.0, 0.0

            RT_row.append(rt_client)
            U_row.append(utility)
            Warm_row.append(warm_rate)
            Cold_row.append(cold_rate)
            Success_row.append(success_rate)
            Queue_row.append(mean_queue)
            Init_row.append(0.0 if np.isnan(mean_init) else mean_init)
            Init_75_row.append(0.0 if np.isnan(init_75) else init_75)
            NetOv_row.append(net_overhead)
            PoolMem_row.append(SYSTEM_MEMORY)
            FuncMem_row.append(MEM_DEMAND)
            Cpus_row.append(CPU_CORES)
            FuncCpu_row.append(CPU_DEMAND)
            QueueLen_row.append(QUEUE_LENGTH)

        RT_matrix.append(RT_row)
        U_matrix.append(U_row)
        Warm_matrix.append(Warm_row)
        Cold_matrix.append(Cold_row)
        Success_matrix.append(Success_row)
        Queue_matrix.append(Queue_row)
        Init_matrix.append(Init_row)
        Init_75_matrix.append(Init_75_row)
        NetOv_matrix.append(NetOv_row)
        PoolMem_matrix.append(PoolMem_row)
        FuncMem_matrix.append(FuncMem_row)
        Cpus_matrix.append(Cpus_row)
        FuncCpu_matrix.append(FuncCpu_row)
        QueueLen_matrix.append(QueueLen_row)

    np.savez(SERV_TRAIN,
             X=X,
             RT=np.array(RT_matrix),
             U=np.array(U_matrix),
             Warm=np.array(Warm_matrix),
             Cold=np.array(Cold_matrix),
             Success=np.array(Success_matrix),
             Queue=np.array(Queue_matrix),
             Init=np.array(Init_matrix),
             Init_75=np.array(Init_75_matrix),
             NetOv=np.array(NetOv_matrix),
             PoolMem=np.array(PoolMem_matrix),
             FuncMem=np.array(FuncMem_matrix),
             Cpus=np.array(Cpus_matrix),
             FuncCpu=np.array(FuncCpu_matrix),
             QueueLen=np.array(QueueLen_matrix))

    print(f"[OK] Dataset salvato in: {SERV_TRAIN} (con 13 Feature per il ML!)")


if __name__ == "__main__":
    main()
