"""
MSJ FCFS Mean Response Time Estimator
======================================
Implements Corollary 4.1 from:
  "The RESET and MARC Techniques, with Application to Multiserver-Job Analysis"
  Grosof, Hong, Harchol-Balter, Scheller-Wolf (2023)

Formula (Corollary 4.1):
    E[T^MSJ] = (1/lambda*) * (1 + Delta(Y_d^SSS, Y^SSS)) / (1 - lambda/lambda*) + O(1)

where the dominant term is computed from the Simplified Saturated System (SSS).
The additive O(1) constant is not computed — the formula is asymptotically exact
as lambda -> lambda*.

Steps follow Appendix C exactly:
  C.1  Solve CTMC balance equations -> time-average steady state Y
  C.2  Compute throughput lambda* = E_Y[mu_{y,.,1}]  (all completions, incl. self-loops)
  C.3  Compute departure-average steady state Y_d
  C.4  Solve for relative completions Delta(y) via linear system
  C.5  Normalise Delta so E_Y[Delta(y)] = 0
  Apply Corollary 4.1.

Key design note on self-loop transitions
-----------------------------------------
In the SSS, a state like [1,1] can transition to itself when the completing job
is replaced by a same-class job.  Self-loops DO contribute to lambda* (completions
happen), but they must NOT appear on the off-diagonal of the CTMC generator Q_gen
(they cancel on both sides of the balance equations).  This implementation handles
this correctly by separating mu_completion (all rates) from mu_offdiag (off-diag only).

"""

import numpy as np
from scipy.linalg import solve
from typing import Any, Dict, List, Tuple
from collections import deque


State = Any


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def msj_mean_response_time(
    states: List[State],
    transitions: Dict[Tuple[State, State], float],
    arrival_rate: float,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Estimate E[T^MSJ] using Corollary 4.1 (dominant asymptotic term).

    Parameters
    ----------
    states : list
        Ordered list of SSS state labels (any hashable).
    transitions : dict
        Keys (from_state, to_state), values = completion rate mu_{y,y',1}.
        Self-loop entries (y -> y) are allowed.
    arrival_rate : float
        Poisson arrival rate lambda.  Must be < lambda*.
    verbose : bool
        Print intermediate quantities if True.

    Returns
    -------
    dict with keys:
        'lambda_star'  : stability threshold
        'Y'            : time-average steady state (numpy array, indexed by states)
        'Y_d'          : departure-average steady state (numpy array)
        'Delta'        : relative completions per state (numpy array)
        'Delta_Y_d'    : E_{Y_d}[Delta(y)]
        'E_T_dominant' : dominant term of mean response time
        'rho'          : lambda / lambda*
    """
    n = len(states)
    idx = {s: i for i, s in enumerate(states)}

    # ------------------------------------------------------------------
    # Build rate matrices
    # ------------------------------------------------------------------
    # mu_completion[i]  total completion rate FROM state i  (includes self-loops)
    #                   used for lambda* and Delta recurrence
    # mu_offdiag[i]     sum of ONLY off-diagonal completion rates from i
    #                   used for CTMC generator diagonal
    # Q_compl[i,j]      rate from i to j  (self-loops on diagonal if present)

    mu_completion = np.zeros(n)
    mu_offdiag    = np.zeros(n)
    Q_compl       = np.zeros((n, n))

    for (src, dst), rate in transitions.items():
        i, j = idx[src], idx[dst]
        Q_compl[i, j] += rate
        mu_completion[i] += rate
        if i != j:
            mu_offdiag[i] += rate

    # ------------------------------------------------------------------
    # Step C.1 — CTMC balance equations -> time-average steady state Y
    # Generator:
    #   Q_gen[i,j] = Q_compl[i,j]          for i != j
    #   Q_gen[i,i] = -mu_offdiag[i]        (self-loops excluded from diagonal)
    # Solve pi @ Q_gen = 0  (row convention),  sum(pi) = 1
    # ------------------------------------------------------------------
    Q_gen = Q_compl.copy()
    for i in range(n):
        Q_gen[i, i] = -mu_offdiag[i]

    # Solve via augmented system: replace last row with normalisation
    A = Q_gen.T.copy()
    b = np.zeros(n)
    A[-1, :] = 1.0
    b[-1] = 1.0
    Y = solve(A, b)
    Y = np.maximum(Y, 0.0)
    Y /= Y.sum()

    # ------------------------------------------------------------------
    # Step C.2 — Throughput / lambda*  (Eq. C.2)
    # lambda* = E_Y[mu_{y,.,1}]  includes all completions (self-loops too)
    # ------------------------------------------------------------------
    lambda_star = float(Y @ mu_completion)

    # ------------------------------------------------------------------
    # Step C.3 — Departure-average steady state Y_d  (Eq. C.3)
    # P(Y_d = y') = (1/lambda*) * sum_y P(Y=y) * mu_{y,y',1}
    # ------------------------------------------------------------------
    Y_d = (Y @ Q_compl) / lambda_star
    Y_d = np.maximum(Y_d, 0.0)
    Y_d /= Y_d.sum()

    # ------------------------------------------------------------------
    # Step C.4 — Relative completions Delta(y)  (Corollary D.1 / Eq. C.4)
    #
    # When all transitions are completions (SSS with exponential durations):
    #   Delta(y) = 1 - lambda*/mu_y + sum_{y'} (mu_{y,y'}/mu_y) * Delta(y')
    # where mu_y = mu_completion[i]  (total rate, including self-loops).
    #
    # Rearranged: (I - P) @ Delta = r
    #   P[i,j] = Q_compl[i,j] / mu_completion[i]  (self-loops on diag included)
    #   r[i]   = 1 - lambda* / mu_completion[i]
    #
    # The system is rank-1 deficient.  Uniqueness via Eq. C.5: Y @ Delta = 0.
    # Replace the last equation of (I-P)Delta = r with Y @ Delta = 0.
    # ------------------------------------------------------------------
    P = np.zeros((n, n))
    r = np.zeros(n)
    for i in range(n):
        if mu_completion[i] > 0:
            P[i, :] = Q_compl[i, :] / mu_completion[i]
            r[i] = 1.0 - lambda_star / mu_completion[i]

    A_delta = np.eye(n) - P
    b_delta = r.copy()
    A_delta[-1, :] = Y      # normalisation constraint
    b_delta[-1] = 0.0

    Delta = solve(A_delta, b_delta)

    # ------------------------------------------------------------------
    # Corollary 4.1  E[T^MSJ] dominant term
    # ------------------------------------------------------------------
    Delta_Y_d = float(Y_d @ Delta)
    rho = arrival_rate / lambda_star

    if rho >= 1.0:
        raise ValueError(
            f"System is unstable: lambda={arrival_rate:.6f} >= lambda*={lambda_star:.6f}. "
            "Corollary 4.1 requires lambda < lambda*."
        )

    E_T_dominant = (1.0 / lambda_star) * (1.0 + Delta_Y_d) / (1.0 - rho)

    if verbose:
        _print_results(states, arrival_rate, lambda_star, rho,
                       Y, Y_d, Delta, Delta_Y_d, E_T_dominant)

    return {
        "lambda_star":   lambda_star,
        "Y":             Y,
        "Y_d":           Y_d,
        "Delta":         Delta,
        "Delta_Y_d":     Delta_Y_d,
        "E_T_dominant":  E_T_dominant,
        "rho":           rho,
    }


def _print_results(states, arrival_rate, lambda_star, rho,
                   Y, Y_d, Delta, Delta_Y_d, E_T_dominant):
    w = 62
    print(f"\n{'='*w}")
    print("MSJ FCFS Mean Response Time  —  Corollary 4.1")
    print(f"{'='*w}")
    print(f"  lambda         = {arrival_rate:.6f}")
    print(f"  lambda*        = {lambda_star:.6f}")
    print(f"  rho = lambda/lambda* = {rho:.6f}")
    print()
    print(f"  {'State':<20} {'Y':>10} {'Y_d':>10} {'Delta':>10}")
    print(f"  {'-'*53}")
    for s, y, yd, d in zip(states, Y, Y_d, Delta):
        print(f"  {str(s):<20} {y:>10.5f} {yd:>10.5f} {d:>10.5f}")
    print()
    print(f"  Delta(Y_d)     = {Delta_Y_d:.6f}")
    print(f"  E[T] dominant  = {E_T_dominant:.6f}")
    print(f"  (Plus an O(1) additive constant not captured here)")
    print(f"{'='*w}\n")



# ---------------------------------------------------------------------------
# General SSS builder via BFS state enumeration
# ---------------------------------------------------------------------------

def build_general_sss(
    k: int,
    job_classes: List[Dict],
) -> Tuple[List[Tuple], Dict[Tuple, float]]:
    """
    Build the SSS for an arbitrary set of job classes with exponential durations.

    Each class dict must have:
        'need' : int   — server need
        'rate' : float — exponential service rate  (= 1 / mean duration)
        'prob' : float — arrival probability  (normalised internally)

    The SSS state is a tuple of server-needs of currently tracked jobs,
    in FCFS order, with total server need of in-service jobs >= k and
    the minimum number of jobs needed to reach that threshold.

    Uses BFS from a canonical initial state to enumerate all reachable states
    and builds the transition rate dict automatically.

    Returns
    -------
    states      : sorted list of reachable SSS states (tuples of ints)
    transitions : dict  (from_state, to_state) -> rate
    """
    probs = np.array([c["prob"] for c in job_classes], dtype=float)
    probs /= probs.sum()
    needs = [int(c["need"]) for c in job_classes]
    rates = [float(c["rate"]) for c in job_classes]
    need_to_cls = {need: i for i, need in enumerate(needs)}

    def jobs_in_service(state: tuple) -> List[int]:
        """Return positions of jobs currently in service (FCFS packing up to k)."""
        serving, total = [], 0
        for pos, need in enumerate(state):
            if total + need <= k:
                total += need
                serving.append(pos)
            else:
                break
        return serving

    def admit_jobs(after_completion: tuple) -> list:
        """
        After one job completes, reconstruct a valid SSS state by admitting
        new jobs from the back until the SSS stopping condition is met.

        SSS stopping condition (Appendix B):
          Stop admitting when:
            (a) total server need of in-service jobs >= k, OR
            (b) a HoL blocker already exists (a job that cannot fit).

        Returns list of (next_state, probability).
        """
        frontier = [(after_completion, 1.0)]
        final = {}

        while frontier:
            state, prob = frontier.pop()
            in_svc_pos = jobs_in_service(state)
            total_in_svc = sum(state[p] for p in in_svc_pos)
            has_blocker = len(in_svc_pos) < len(state)

            if total_in_svc >= k or has_blocker:
                # SSS state is complete
                final[state] = final.get(state, 0.0) + prob
            else:
                # Draw one more job
                for ci in range(len(needs)):
                    extended = state + (needs[ci],)
                    frontier.append((extended, prob * probs[ci]))

        return list(final.items())

    def initial_state() -> tuple:
        """Canonical seed: fill with class-0 jobs until need >= k."""
        state, total = [], 0
        while total < k:
            state.append(needs[0])
            total += needs[0]
        return tuple(state)

    visited: set = set()
    queue = deque()
    trans: Dict = {}

    seed = initial_state()
    queue.append(seed)
    visited.add(seed)

    while queue:
        state = queue.popleft()
        for pos in jobs_in_service(state):
            need = state[pos]
            ci = need_to_cls.get(need)
            if ci is None:
                continue
            rate = rates[ci]
            remaining = state[:pos] + state[pos + 1:]
            for next_state, prob in admit_jobs(remaining):
                key = (state, next_state)
                trans[key] = trans.get(key, 0.0) + rate * prob
                if next_state not in visited:
                    visited.add(next_state)
                    queue.append(next_state)

    states = sorted(visited)
    return states, trans




def compute_approx_rt (arv_rates, srv_times, mem_demands: List[int], total_memory: int, verbose=False):
    n = len(arv_rates)
    assert(n == len(srv_times))
    assert(n == len(mem_demands))
    assert(total_memory >= 1)

    total_arv_rate = sum(arv_rates)

    classes = []
    for i in range(n):
        classes.append( {"need": mem_demands[i], "rate": 1/srv_times[i], "prob": arv_rates[i]/total_arv_rate})

    states, transitions = build_general_sss(k=total_memory, job_classes=classes)
    res = msj_mean_response_time(states, transitions, arrival_rate=total_arv_rate)
    if verbose:
        print(res)
    return res["E_T_dominant"]

def test_mm1():
    mu = 1/0.33
    lamb = 0.2
    rt = compute_approx_rt([lamb, lamb], [1/mu, 2/mu], [1, 1], 1)
    print(rt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_mm1()
