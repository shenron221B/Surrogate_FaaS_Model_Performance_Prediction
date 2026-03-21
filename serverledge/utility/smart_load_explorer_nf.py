import random


class SmartLoadExplorerNF:
    def __init__(self, config_dict):
        self.n = len(config_dict["func_names"])
        self.config = config_dict

        # stati: NORMAL, CRITICAL, STOPPED
        self.states = ["NORMAL"] * self.n
        self.current_loads = list(self.config["start_loads"])
        self.prev_loads = list(self.config["start_loads"])

        self.crit_start_loads = [None] * self.n
        self.stop_loads = [None] * self.n

        self.current_steps = list(self.config["step_sizes"])
        self.current_decay_steps = list(self.config["decay_step_sizes"])

    def get_next_load(self):
        # se tutte le funzioni sono STOPPED, l'esplorazione globale è finita
        if all(s == "STOPPED" for s in self.states):
            return None

        next_loads = []
        for i in range(self.n):
            if self.states[i] == "NORMAL":
                load = self.current_loads[i] + self.current_steps[i]
                self.current_steps[i] += self.current_steps[i] * self.config["step_inc_percs"][i]

            elif self.states[i] == "CRITICAL":
                load = self.current_loads[i] + self.current_decay_steps[i]

            elif self.states[i] == "STOPPED":
                # generazione carico casuale tra l'inizio dello stop per una funzione e lo stop globale
                min_l = self.crit_start_loads[i] if self.crit_start_loads[i] else self.config["start_loads"][i]
                max_l = self.stop_loads[i]
                load = round(random.uniform(min_l, max_l), 3)

            next_loads.append(round(load, 3))

        self.prev_loads = list(self.current_loads)
        self.current_loads = next_loads
        return next_loads

    def report_result(self, loads, utilities, fail_rates):
        for i in range(self.n):
            if self.states[i] == "STOPPED":
                continue

            u = utilities[i]
            d = fail_rates[i]

            # controllo STOP
            stop_met = self.config["stop_metrics"][i]
            is_stop = False

            if stop_met == "utility_only" and u <= self.config["stop_util_limits"][i]:
                is_stop = True
            elif stop_met == "drop_rate_only" and d >= self.config["stop_drop_limits"][i]:
                is_stop = True
            elif stop_met in ["utility_or_drop_rate", "drop_rate_or_utility"]:
                if u <= self.config["stop_util_limits"][i] or d >= self.config["stop_drop_limits"][i]:
                    is_stop = True
            elif stop_met in ["utility_and_drop_rate", "drop_rate_and_utility"]:
                if u <= self.config["stop_util_limits"][i] and d >= self.config["stop_drop_limits"][i]:
                    is_stop = True

            if is_stop:
                self.states[i] = "STOPPED"
                self.stop_loads[i] = loads[i]
                print(f"  -> [F{i + 1}] STOP Raggiunto! Passo a generazione Random Wait.")
                continue

            # controllo CRITICAL e BACKTRACKING
            if self.states[i] == "NORMAL":
                dec_met = self.config["decay_metrics"][i]
                is_crit = False

                if dec_met == "utility_only" and u <= self.config["decay_util_threshs"][i]:
                    is_crit = True
                elif dec_met == "drop_rate_only" and d >= self.config["decay_drop_threshs"][i]:
                    is_crit = True
                elif dec_met in ["utility_or_drop_rate", "drop_rate_or_utility"]:
                    if u <= self.config["decay_util_threshs"][i] or d >= self.config["decay_drop_threshs"][i]:
                        is_crit = True
                elif dec_met in ["utility_and_drop_rate", "drop_rate_and_utility"]:
                    if u <= self.config["decay_util_threshs"][i] and d >= self.config["decay_drop_threshs"][i]:
                        is_crit = True

                if is_crit:
                    self.states[i] = "CRITICAL"
                    self.crit_start_loads[i] = self.prev_loads[i]
                    self.current_loads[i] = self.prev_loads[i]
                    print(
                        f"  -> [F{i + 1}] Fascia CRITICA. Backtracking attivato da {loads[i]} a {self.prev_loads[i]}.")
                    continue