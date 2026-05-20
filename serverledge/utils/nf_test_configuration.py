
CONFIG_MATRICES = {
    "test_3f_mix": {
        "func_names": ["hash_worker", "matrix_mem", "pi_calc"],
        "mem_reqs": [256, 256, 256],
        "start_loads": [0.5, 0.5, 0.5],  # carico di partenza
        "step_sizes": [1.0, 1.0, 1.0],  # step iniziale
        "step_inc_percs": [0.25, 0.25, 0.25],  # inc. % step iniziale

        "decay_metrics": ["utility_only", "utility_only", "utility_only"],  # metrica decadimento
        "decay_util_threshs": [0.80, 0.80, 0.80],  # soglia utils decadimento
        "decay_drop_threshs": [1.0, 1.0, 1.0],  # soglia drop rate decadimento

        "decay_step_sizes": [0.5, 0.5, 0.5],  # step critico
        "decay_step_inc": [0.25, 0.25, 0.25],  # inc. % step critico
        "decay_step_dec": [0.25, 0.25, 0.25],  # rallentamento step critico
        "delta_lower": [0.05, 0.05, 0.05],  # delta lower
        "delta_upper": [0.10, 0.10, 0.10],  # delta upper

        "stop_metrics": ["drop_rate_or_utility", "drop_rate_or_utility", "drop_rate_or_utility"],  # metrica di stop
        "stop_util_limits": [0.05, 0.05, 0.05],  # soglia di stop utils
        "stop_drop_limits": [0.95, 0.95, 0.95]  # soglia di stop drop rate
    },

    "test_2f_weather_pi": {
        "func_names": ["weather", "pi_calc"],
        "mem_reqs": [256, 256],

        "start_loads": [0.1, 0.1],
        "step_sizes": [0.4, 0.4],
        "step_inc_percs": [0.1, 0.1],

        "decay_metrics": ["utility_only", "utility_only"],
        "decay_util_threshs": [0.85, 0.85],
        "decay_drop_threshs": [1.0, 1.0],

        "decay_step_sizes": [0.2, 0.2],
        "decay_step_inc": [0.15, 0.25],
        "decay_step_dec": [0.25, 0.25],
        "delta_lower": [0.05, 0.05],
        "delta_upper": [0.10, 0.10],

        "stop_metrics": ["utility_or_drop_rate", "utility_or_drop_rate"],
        "stop_util_limits": [0.05, 0.05],
        "stop_drop_limits": [0.90, 0.90]
    },

    "test_2f_matrix_hash": {
        "func_names": ["matrix_mem", "hash_worker"],
        "mem_reqs": [256, 256],

        "start_loads": [0.01, 0.01],
        "step_sizes": [0.5, 0.5],
        "step_inc_percs": [0.1, 0.1],

        "decay_metrics": ["utility_only", "utility_only"],
        "decay_util_threshs": [0.80, 0.80],
        "decay_drop_threshs": [1.0, 1.0],

        "decay_step_sizes": [0.2, 0.2],
        "decay_step_inc": [0.25, 0.25],
        "decay_step_dec": [0.25, 0.25],
        "delta_lower": [0.05, 0.05],
        "delta_upper": [0.10, 0.10],

        "stop_metrics": ["utility_or_drop_rate", "utility_or_drop_rate"],
        "stop_util_limits": [0.05, 0.05],
        "stop_drop_limits": [0.90, 0.90]
    }
}

