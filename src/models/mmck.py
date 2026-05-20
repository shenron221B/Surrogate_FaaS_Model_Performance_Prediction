# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------

import numpy as np
from scipy.linalg import expm

def mmck (l, mu, c, K):
    def compute_state_probabilities(lambda_, mu, s, k):
        from scipy.special import gammaln  # for log(n!) = gammaln(n + 1)
        """
        Log-space computation of steady-state probabilities π_n
        for an M/M/s/k queue, numerically stable even for large k.
        """
        rho = lambda_ / mu
        log_pi = np.zeros(k + 1)

        for n in range(k + 1):
            if n < s:
                log_pi[n] = n * np.log(rho) - gammaln(n + 1)
            else:
                log_pi[n] = n * np.log(rho) - gammaln(s + 1) - (n - s) * np.log(s)

        # Normalize in log-space
        max_log_pi = np.max(log_pi)  # for numerical stability
        log_sum = max_log_pi + np.log(np.sum(np.exp(log_pi - max_log_pi)))
        pi = np.exp(log_pi - log_sum)

        return pi
    assert(c > 0)
    assert(K >= c)
    assert(mu > 0)
    assert(l > 0)

    rho=l/mu

    pi = compute_state_probabilities(l, mu, c, K)
    pBlock = pi[K]

    # Avg time in the system
    Lq = sum([(k-c)*pi[k] for k in range(c,K+1) ])
    Wq = Lq / (l*(1-pBlock))
    RT = 1/mu + Wq

    def phase_type_cdf(t, rates):
        """
        Computes the CDF of a sum of exponentials (PH distribution),
        even when rates are repeated, using matrix exponentials.

        Parameters:
        - t: time at which to evaluate the CDF
        - rates: list of rates [λ1, λ2, ..., λn] (can be repeated)

        Returns:
        - CDF value at time t
        """
        n = len(rates)
        # Generator matrix T for sequential phases
        T = np.zeros((n, n))
        for i in range(n):
            T[i, i] = -rates[i]
            if i < n - 1:
                T[i, i+1] = rates[i]
        # Initial state: start in phase 0
        alpha = np.zeros(n)
        alpha[0] = 1.0
        # Matrix exponential
        exp_Tt = expm(T * t)
        one_vec = np.ones(n)
        survival_prob = alpha @ exp_Tt @ one_vec
        return 1.0 - survival_prob

    # prob rT <= t
    def rt_cdf(t):
        """
        Compute the CDF of the response time in an M/M/s/k queue.

        Parameters:
        - t: time at which to evaluate the CDF
        - lambda_: arrival rate
        - mu: service rate
        - s: number of servers
        - k: total system capacity (including service + waiting)

        Returns:
        - Response time CDF at time t
        """
        # Conditional state probabilities given the job is admitted
        admitted_states = range(K)  # Only up to k-1 are admitted
        p_arrival = pi[:K] / np.sum(pi[:K])

        total_cdf = 0.0
        for n in admitted_states:
            prob = p_arrival[n]
            if n < c:
                # Immediate service
                total_cdf += prob * (1 - np.exp(-mu * t))
            else:
                m = n - c + 1  # waiting stages
                rates = [c * mu] * m + [mu]
                total_cdf += prob * phase_type_cdf(t, rates)

        return total_cdf


    return RT, pBlock, pi[0], rt_cdf

# K = maximum capacity = c + queue length
#jdef mmck (l, mu, c, K):
#j    assert(c > 0)
#j    assert(K >= c)
#j    assert(mu > 0)
#j    assert(l > 0)
#j    cfact = math.factorial(c)
#j
#j    #v=0
#j    #for k in range(0, c+1):
#j    #    v += l**k/(mu**k * math.factorial(k))
#j    #v2=0
#j    #for k in range(c+1, K+1):
#j    #    v2 += l**(k-c)/(mu**(k-c)*c**(k-c))
#j    #v2 *= l**c/(mu**c * cfact)
#j    #p0 = 1.0/(v + v2)
#j
#j    #pk = lambda k: (l/mu)**k/(c**(k-c)*cfact)*p0
#j    rho=l/mu
#j
#j    def P(n):
#j        if n < c:
#j            return (rho ** n) / math.factorial(n)
#j        else:
#j            return (rho ** n) / (cfact * (c ** (n - c)))
#j
#j    normalizer = sum(P(n) for n in range(K + 1))
#j    pi = np.zeros(K + 1)
#j    pi[0] = 1.0 / normalizer
#j
#j    for n in range(1, K + 1):
#j        if n < c:
#j            pi[n] = pi[0] * (rho ** n) / math.factorial(n)
#j        else:
#j            pi[n] = pi[0] * (rho ** n) / (cfact * (c ** (n - c)))
#j
#j    pBlock = pi[K]
#j
#j    # Avg time in the system
#j    Lq = sum([(k-c)*pi[k] for k in range(c,K+1) ])
#j    Wq = Lq / (l*(1-pBlock))
#j    RT = 1/mu + Wq
#j
#j
#j    def phase_type_cdf(t, rates):
#j        """
#j        Computes the CDF of a sum of exponentials (PH distribution),
#j        even when rates are repeated, using matrix exponentials.
#j
#j        Parameters:
#j        - t: time at which to evaluate the CDF
#j        - rates: list of rates [λ1, λ2, ..., λn] (can be repeated)
#j
#j        Returns:
#j        - CDF value at time t
#j        """
#j        n = len(rates)
#j        # Generator matrix T for sequential phases
#j        T = np.zeros((n, n))
#j        for i in range(n):
#j            T[i, i] = -rates[i]
#j            if i < n - 1:
#j                T[i, i+1] = rates[i]
#j        # Initial state: start in phase 0
#j        alpha = np.zeros(n)
#j        alpha[0] = 1.0
#j        # Matrix exponential
#j        exp_Tt = expm(T * t)
#j        one_vec = np.ones(n)
#j        survival_prob = alpha @ exp_Tt @ one_vec
#j        return 1.0 - survival_prob
#j
#j    # prob rT <= t
#j    def rt_cdf(t):
#j        """
#j        Compute the CDF of the response time in an M/M/s/k queue.
#j
#j        Parameters:
#j        - t: time at which to evaluate the CDF
#j        - lambda_: arrival rate
#j        - mu: service rate
#j        - s: number of servers
#j        - k: total system capacity (including service + waiting)
#j
#j        Returns:
#j        - Response time CDF at time t
#j        """
#j        # Conditional state probabilities given the job is admitted
#j        admitted_states = range(K)  # Only up to k-1 are admitted
#j        p_arrival = pi[:K] / np.sum(pi[:K])
#j
#j        total_cdf = 0.0
#j        for n in admitted_states:
#j            prob = p_arrival[n]
#j            if n < c:
#j                # Immediate service
#j                total_cdf += prob * (1 - np.exp(-mu * t))
#j            else:
#j                m = n - c + 1  # waiting stages
#j                rates = [c * mu] * m + [mu]
#j                total_cdf += prob * phase_type_cdf(t, rates)
#j
#j        return total_cdf
#j
#j
#j    return RT, pBlock, pi[0], rt_cdf

class MMckModel:

    def __init__ (self, model, partitioning_mode_alternative=False):
        self.model = model

        n = len(model.serv_times)

        # Greedy partinioning: TODO
        servers = np.zeros(n).astype(int)
        queue_cap = np.zeros(n).astype(int)
        avail_memory = model.memory

        flows = list(range(n))
        flows = sorted(flows, key = lambda x: float(model.mem_demands[x]), reverse=True)


        if not partitioning_mode_alternative:
            while avail_memory >= np.min(model.mem_demands):
                for f in flows:
                    while avail_memory >= model.mem_demands[f]:
                        servers[f] += 1
                        avail_memory -= model.mem_demands[f]
        else:
            while avail_memory >= np.min(model.mem_demands):
                for f in flows:
                    if avail_memory >= model.mem_demands[f]:
                        servers[f] += 1
                        avail_memory -= model.mem_demands[f]
                        
        self.queue_cap = np.floor(model.queue_capacity/(n*np.ones(n))).astype(int)
        self.servers = servers
        self.N = n

        allocated_memory = sum(servers*model.mem_demands)/model.memory*100

    def predict (self, X):
        RT = np.zeros(X.shape)
        U = np.zeros(X.shape)

        for i in range(self.N):
            mu = 1.0/self.model.serv_times[i]
            c = self.servers[i]
            K = c + self.queue_cap[i]
            if c == 0:
                # no resources
                continue # TODO
            for j in range(X.shape[0]):
                rt, pBlock, _, rtCDF = mmck(X[j,i], mu, c, K)
                RT[j,i] = rt
                U[j,i] = rtCDF(self.model.deadlines[i])*(1-pBlock)

        return RT, U

    def predict_rt (self, X):
        RT = np.zeros(X.shape)
        for i in range(self.N):
            mu = 1.0/self.model.serv_times[i]
            c = self.servers[i]
            K = c + self.queue_cap[i]
            for j in range(X.shape[0]):
                rt, pBlock, _, _ = mmck(X[j,i], mu, c, K)
                RT[j,i] = rt
        return RT


if __name__ == "__main__":
    c=20
    for K in range(c,50):
        print(K)
        r, _, _, cdf = mmck(1, 1.4, c, K)
        for t in range(1,100):
            print(cdf(t*0.01))
        print("----")
#    import model
#    import numpy as np
#
#    def create_random_X (rng, model, samples, max_rho=1.1):
#        n = len(model.serv_times)
#        max_lambda = max_rho*model.memory/np.array(model.serv_times)/np.array(model.mem_demands)/n
#        min_lambda = 0.005/np.array(model.serv_times)/n
#
#        X = np.zeros((samples, n))
#        for i in range(n):
#            X[:,i] = rng.uniform(min_lambda[i], max_lambda[i], samples)
#        return X
#
#    rng = np.random.default_rng()
#    m = model.random_model(rng, queue_cap=10)
#    m.queue_capacity = 10
#    X = create_random_X(rng, m, 10, 0.8)
#
#    k = MMckModel(m)
#    print(k.predict(X))

