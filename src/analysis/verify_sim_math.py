import sys
import os
import copy
from models.model import model_from_conf
from data.utils import go_simulate
import yaml


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 verify_sim_math.py <dir_configurazione>")
        sys.exit(1)

    conf_file = os.path.join(sys.argv[1], "simulator-conf.yml")
    with open(conf_file, 'r') as f:
        config_data = yaml.safe_load(f)

    base_model = model_from_conf(config_data)
    num_funcs = len(base_model.mem_demands)

    test_rates = [
        [2.0] * num_funcs,
        [15.0] * num_funcs
    ]

    models_to_simulate = []
    for rates in test_rates:
        m = copy.deepcopy(base_model)
        m.arv_rates = rates
        models_to_simulate.append(m)

    print("\nLancio simulazione per verifica matematica...")
    results = go_simulate(models_to_simulate, n_arrivals=100000, seed=42, parallelism=1)

    print("\n" + "=" * 60)
    print(" VERIFICA MATEMATICA SIMULATORE GO")
    print("=" * 60)

    for i, res in enumerate(results):
        target_rate = test_rates[i]
        sim_time = res["Time"]
        arrivals = res["Arrivals"]
        completions = res["Completions"]

        print(f"\n--- SCENARIO {i + 1} | Target Arrivi: {target_rate} req/s ---")
        print(f"Tempo totale simulato : {sim_time:.2f} secondi")

        for f in range(num_funcs):
            eff_rate = arrivals[f] / sim_time if sim_time > 0 else 0
            drop_count = arrivals[f] - completions[f]

            print(f"  Funzione {f + 1}:")
            print(f"    - Arrivi totali   : {arrivals[f]}")
            print(f"    - Completamenti   : {completions[f]}")
            print(f"    - Richieste perse : {drop_count} (Coda piena)")
            print(f"    - Tasso Effettivo : {eff_rate:.3f} req/s (Target: {target_rate[f]:.3f})")


if __name__ == "__main__":
    main()