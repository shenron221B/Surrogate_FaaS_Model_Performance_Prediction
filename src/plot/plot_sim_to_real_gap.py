import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def print_usage():
    print("Uso: python3 plot_sim_to_real_gap.py <matrice_2f> <matrice_3f> <matrice_5f>")
    sys.exit(1)

def extract_metrics(npz_path):
    data = np.load(npz_path)
    keys_to_check = ['X', 'Success', 'RT', 'Queue', 'U', 'Cold']
    min_len = min([data[k].shape[0] for k in keys_to_check if k in data.files])

    X_cut = data['X'][:min_len]
    Success_cut = data['Success'][:min_len]
    RT_cut = data['RT'][:min_len]
    Queue_cut = data['Queue'][:min_len]
    U_cut = data['U'][:min_len]
    Cold_cut = data['Cold'][:min_len]

    throughput_matrix = Success_cut * X_cut
    scenario_throughput = np.sum(throughput_matrix, axis=1)

    metrics = {
        'RT': np.mean(RT_cut),
        'Queue': np.mean(Queue_cut),
        'Utility': np.mean(U_cut) * 100,
        'Throughput': np.mean(scenario_throughput),
        'ColdStart': np.mean(Cold_cut) * 100
    }
    return metrics

def process_directory(dir_path):
    real_path = os.path.join(dir_path, "dataset.npz")
    sim_files = glob.glob(os.path.join(dir_path, "dataset_simulated_*.npz"))
    if not os.path.exists(real_path) or not sim_files:
        sys.exit(1)
    sim_path = sim_files[0]
    real_metrics = extract_metrics(real_path)
    sim_metrics = extract_metrics(sim_path)
    return real_metrics, sim_metrics

def plot_combined_metric_bar(data, metric_key, config, save_path):
    funcs = [2, 3, 5]
    qlens = [0, 5, 15]
    x_positions = np.array([0, 1, 2,  4, 5, 6,  8, 9, 10]) 
    labels = []
    real_vals, sim_vals = [], []
    
    for f in funcs:
        for q in qlens:
            labels.append(f"Coda: {q}")
            real_vals.append(data[f][q]['real'][metric_key])
            sim_vals.append(data[f][q]['sim'][metric_key])

    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 7))
    color_sim, color_real = config['colors']
    
    ax.bar(x_positions - width/2, sim_vals, width, label='Simulatore', color=color_sim)
    ax.bar(x_positions + width/2, real_vals, width, label='Sistema Reale', color=color_real)

    ax.set_ylabel(config['y_bar'], fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    y_offset = -0.08
    ax.text(1, y_offset, 'Funzioni: 2', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=18, fontweight='normal')
    ax.text(5, y_offset, 'Funzioni: 3', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=18, fontweight='normal')
    ax.text(9, y_offset, 'Funzioni: 5', ha='center', va='center', transform=ax.get_xaxis_transform(), fontsize=18, fontweight='normal')

    ax.legend(fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplots_adjust(bottom=0.15)
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_combined_gap_bar(data, func_count, metrics_order, metrics_config, save_path):
    qlens = [0, 5, 15]
    x_positions = np.arange(len(qlens))
    fig, ax = plt.subplots(figsize=(12, 7))
    
    num_metrics = len(metrics_order)
    bar_width = 0.15
    spacing = 0.02
    total_group_width = (bar_width * num_metrics) + (spacing * (num_metrics - 1))
    start_offset = -total_group_width / 2 + bar_width / 2
    
    max_height = 0 
    
    for i, m_key in enumerate(metrics_order):
        gaps = [data[func_count][q]['gap'][m_key] for q in qlens]
        if gaps: 
            max_height = max(max_height, max(gaps))
        
        offset = start_offset + i * (bar_width + spacing)
        bars = ax.bar(x_positions + offset, gaps, bar_width, color='#ff7f0e')
        
        for bar in bars:
            height = bar.get_height()
            # Testo verticale ENORME
            ax.text(bar.get_x() + bar.get_width()/2., height + (max_height * 0.02),
                    metrics_config[m_key]['short_name'],
                    ha='center', va='bottom', rotation=90, fontsize=16, fontweight='normal')

    ax.set_xlabel('Capacità della coda', fontsize=18)
    ax.set_ylabel('Gap percentuale assoluto (%)', fontsize=18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(q) for q in qlens], fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    
    ax.set_ylim(bottom=0, top=max_height * 1.35)
    ax.axhline(0, color='black', linewidth=1)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    if len(sys.argv) < 4:
        print_usage()

    base_results_dir = "/root/tesi_project/serverledge/results"
    
    mat_2f = sys.argv[1]
    mat_3f = sys.argv[2]
    mat_5f = sys.argv[3]
    matrix_map = {2: mat_2f, 3: mat_3f, 5: mat_5f}
    
    funcs = [2, 3, 5]
    qlens = [0, 5, 15]

    metrics_config = {
        'RT':         {'short_name': 'RT Medio', 'y_bar': 'Secondi (s)', 'is_abs_gap': False, 'colors': ('#6baed6', '#08519c')}, 
        'Queue':      {'short_name': 'Queue Time', 'y_bar': 'Secondi (s)', 'is_abs_gap': False, 'colors': ('#bcbddc', '#54278f')}, 
        'Utility':    {'short_name': 'Utilità', 'y_bar': 'Prob. Deadline (%)', 'is_abs_gap': True, 'colors': ('#a1d99b', '#006d2c')}, 
        'Throughput': {'short_name': 'Throughput', 'y_bar': 'Throughput (req/s)', 'is_abs_gap': False, 'colors': ('#fb6a4a', '#cb181d')}, 
        'ColdStart':  {'short_name': 'Cold Start', 'y_bar': 'Prob. Cold Start (%)', 'is_abs_gap': True, 'colors': ('#dcb38a', '#8c510a')} 
    }

    metrics_gap_order = ['RT', 'Utility', 'ColdStart', 'Queue'] 

    print("\n[INFO] Esplorazione cartelle in corso...")
    data = {f: {q: {'real': {}, 'sim': {}, 'gap': {}} for q in qlens} for f in funcs}
    
    for f in funcs:
        for q in qlens:
            mat_name = matrix_map[f]
            dir_name = f"{f}f2GBpm_qlen{q}_8core_256"
            full_path = os.path.join(base_results_dir, dir_name, mat_name)
            
            r_met, s_met = process_directory(full_path)
            data[f][q]['real'] = r_met
            data[f][q]['sim'] = s_met
            
            for m_key, conf in metrics_config.items():
                r_val = r_met[m_key]
                s_val = s_met[m_key]
                if conf['is_abs_gap']:
                    gap = abs(s_val - r_val)
                else:
                    gap = abs(((s_val - r_val) / r_val * 100)) if r_val > 0 else 0
                data[f][q]['gap'][m_key] = gap

    base_out = os.path.join(base_results_dir, "gap_comparison")
    out_metrics = os.path.join(base_out, "metriche_raggruppate")
    out_gaps = os.path.join(base_out, "gaps_raggruppati")
    
    os.makedirs(out_metrics, exist_ok=True)
    os.makedirs(out_gaps, exist_ok=True)
    
    for m_key, conf in metrics_config.items():
        save_path = os.path.join(out_metrics, f"{m_key}_all_scenarios.png")
        plot_combined_metric_bar(data, m_key, conf, save_path)

    for f in funcs:
        save_path = os.path.join(out_gaps, f"gap_percentage_{f}f.png")
        plot_combined_gap_bar(data, f, metrics_gap_order, metrics_config, save_path)

    print("\n[OK] Generazione completata con successo!")

if __name__ == "__main__":
    main()
