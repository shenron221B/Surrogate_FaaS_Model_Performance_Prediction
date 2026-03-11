class SmartLoadExplorer:
    def __init__(self, num_funcs, min_load, initial_step_size, initial_step_inc_perc,
                 decay_metric, decay_drop_thresh, decay_util_thresh,
                 critical_step_size, crit_step_accel_perc, crit_step_decel_perc,
                 crit_diff_lower_bound, crit_diff_upper_bound,
                 stop_metric, stop_drop_limit, stop_util_limit):

        self.num_funcs = num_funcs
        self.current_load = min_load

        # parametri fase iniziale
        self.initial_step_size = initial_step_size
        self.initial_step_inc_perc = initial_step_inc_perc

        # parametri transizione
        self.decay_metric = decay_metric.strip().lower()
        self.decay_drop_thresh = decay_drop_thresh
        self.decay_util_thresh = decay_util_thresh

        # parametri fase critica
        self.critical_step_size = critical_step_size
        self.crit_step_accel_perc = crit_step_accel_perc
        self.crit_step_decel_perc = crit_step_decel_perc
        self.crit_diff_lower_bound = crit_diff_lower_bound
        self.crit_diff_upper_bound = crit_diff_upper_bound

        # parametri stop
        self.stop_metric = stop_metric.strip().lower()
        self.stop_drop_limit = stop_drop_limit
        self.stop_util_limit = stop_util_limit

        # variabili di stato
        self.phase = "NORMAL"  # fasi: NORMAL -> CRITICAL -> DONE
        self.prev_utility = None
        self.prev_drop = None
        self.history = []

        print("\n" + "=" * 50)
        print("[SMART EXPLORER] Inizializzato con logica Feedback-Driven")
        print(f" - Decay Metric: {self.decay_metric} | Stop Metric: {self.stop_metric}")
        print("=" * 50 + "\n")

    def _is_condition_met(self, metric_type, curr_u, curr_d, u_thresh, d_thresh):
        """Helper per verificare se le soglie (Decay o Stop) sono state infrante."""
        drop_met = (curr_d >= d_thresh)
        util_met = (curr_u <= u_thresh)

        if metric_type == "drop_rate_only":
            return drop_met
        elif metric_type == "utility_only":
            return util_met
        elif metric_type == "drop_rate_or_utility":
            return drop_met or util_met
        elif metric_type == "drop_rate_and_utility":
            return drop_met and util_met
        return False

    def _get_metric_delta(self, curr_u, curr_d):
        """Helper per calcolare il degrado rispetto al passo precedente."""
        if self.prev_utility is None or self.prev_drop is None:
            return 0.0

        # calcolo della variazione assoluta del degrado (valori positivi = situazione peggiorata)
        delta_u = self.prev_utility - curr_u  # es: prima 0.90, ora 0.85 -> delta = +0.05
        delta_d = curr_d - self.prev_drop  # es: prima 0.05, ora 0.15 -> delta = +0.10

        # ritorna la metrica di interesse in base alla configurazione
        if self.decay_metric == "drop_rate_only":
            return delta_d
        elif self.decay_metric == "utility_only":
            return delta_u
        else:
            return max(delta_u, delta_d)  # in caso AND/OR si guarda il peggioramento più grave

    def get_next_load(self):
        """Restituisce il carico da testare."""
        if self.phase == "DONE": return None
        return [round(self.current_load, 3)] * self.num_funcs

    def report_result(self, load_tested, utility, drop_rate):
        """Riceve i risultati, calcola i delta e aggiusta i parametri interni."""
        actual_load = load_tested[0]
        delta = self._get_metric_delta(utility, drop_rate)

        print(f"  -> [FEEDBACK] Testato: {actual_load:.3f} | Utility: {utility:.3f} | Drop: {drop_rate * 100:.1f}% | Degrado(\u0394): {delta:.3f}")

        self.history.append((actual_load, utility, drop_rate))

        # 1. verifica condizione di stop
        if self._is_condition_met(self.stop_metric, utility, drop_rate, self.stop_util_limit, self.stop_drop_limit):
            print(f"[SMART EXPLORER] Metrica di STOP raggiunta ({self.stop_metric}). Termino esplorazione.")
            self.phase = "DONE"
            return

        # 2. verifica transizione da NORMAL a CRITICAL
        if self.phase == "NORMAL":
            if self._is_condition_met(self.decay_metric, utility, drop_rate, self.decay_util_thresh, self.decay_drop_thresh):
                print(f"[SMART EXPLORER] Entro in fascia CRITICA. Metrica {self.decay_metric} oltre la soglia.")
                self.phase = "CRITICAL"
            else:
                # incremento dinamico step size normale
                self.current_load += self.initial_step_size
                print(f"  -> [NORMAL] Prossimo salto: +{self.initial_step_size:.3f}")
                self.initial_step_size *= (1.0 + self.initial_step_inc_perc)

        # 3. gestione fascia critica
        if self.phase == "CRITICAL":
            # aggiustamento dinamico del passo critico basato sul delta
            if delta < self.crit_diff_lower_bound:
                # degrado troppo lento -> acceleriamo
                vecchio_step = self.critical_step_size
                self.critical_step_size *= (1.0 + self.crit_step_accel_perc)
                print(f"  -> [CRITICAL] \u0394 ({delta:.3f}) < Lower Bound ({self.crit_diff_lower_bound}). Accelero: step {vecchio_step:.3f} -> {self.critical_step_size:.3f}")
            elif delta > self.crit_diff_upper_bound:
                # degrado troppo brutale -> rallentiamo
                vecchio_step = self.critical_step_size
                self.critical_step_size *= (1.0 - self.crit_step_decel_perc)
                # impostiamo un limite minimo di sicurezza per non bloccarsi
                self.critical_step_size = max(0.01, self.critical_step_size)
                print(f"  -> [CRITICAL] \u0394 ({delta:.3f}) > Upper Bound ({self.crit_diff_upper_bound}). Rallento: step {vecchio_step:.3f} -> {self.critical_step_size:.3f}")
            else:
                print(f"  -> [CRITICAL] \u0394 ({delta:.3f}) nel range ottimale. Mantengo step {self.critical_step_size:.3f}")

            self.current_load += self.critical_step_size

        # aggiornamento dello stato precedente per il calcolo dei delta al prossimo giro
        self.prev_utility = utility
        self.prev_drop = drop_rate