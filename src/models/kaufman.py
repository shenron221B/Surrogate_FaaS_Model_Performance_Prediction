# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model




def _kaufman (model, lambdas):
    M = int(model.memory)
    alpha = np.zeros(len(model.mem_demands))
    for i in range(len(model.mem_demands)):
        alpha[i] = lambdas[i]*model.serv_times[i]

    q = np.zeros(M+1)
    q[0] = 1
    for j in range(1, M+1):
        for i,m in enumerate(model.mem_demands):
            if j-m < 0:
                continue
            q[j] += q[j-m] * m * alpha[i]
        q[j] /= j


    G = np.sum(q)

    bp_per_fun = np.zeros(len(model.mem_demands))
    for i,m in enumerate(model.mem_demands):
        for j in range(0, m):
            if M-j >= 0:
                bp_per_fun[i] += q[M - j]
    bp_per_fun /= G
    return bp_per_fun

class ExactKaufman:

    def __init__ (self, model):
        self.N=len(model.serv_times)
        self.model = model

    def predict (self, X):
        return self.predict_rt(X), self.predict_utility(X)

    def predict_rt (self, X):
        return np.array(self.model.serv_times)*np.ones(X.shape)

    def predict_utility (self, X):
        pb = np.zeros((X.shape[0],self.N))
        for i in range(X.shape[0]):
            pb[i,:] = _kaufman(self.model, X[i,:])
            assert(np.all(pb[i,:] <= 1.0))
            assert(np.all(pb[i,:] >= 0.0))
        deadline_sat = (1.0 - np.exp(-np.array(self.model.deadlines)/np.array(self.model.serv_times)))
        return deadline_sat*(1.0 - pb)

class ApproximateKaufman:

    def __init__ (self, model, X, deg=2):
        N=len(model.serv_times)
        Ntrain=X.shape[0]
    
        Y = np.zeros((Ntrain,N))
        for i in range(Ntrain):
            Y[i,:] = _kaufman(model, X[i,:])
    
        poly = PolynomialFeatures(degree=deg, include_bias=True) 
        X2 = poly.fit_transform(X)
    
        # Create linear regression object
        regr = linear_model.LinearRegression()
        regr.fit(X2, Y)
    
        def utility_predictor (x):
            return (1.0 - np.exp(-np.array(model.deadlines)/np.array(model.serv_times)))*(1.0 - regr.predict(poly.transform(x)))

        def rt_predictor (x):
            return np.array(model.serv_times)*np.ones(x.shape)

        self.util_predictor = utility_predictor
        self.rt_predictor = rt_predictor

    def __call__ (self, x):
        return self.predict(x)

    def predict (self, x):
        return self.rt_predictor(x), self.util_predictor(x)

    def predict_utility (self, x):
        return self.util_predictor(x)


if __name__ == "__main__":
    import model
    import numpy as np

    def create_random_X (rng, model, samples, max_rho=1.1):
        n = len(model.serv_times)
        max_lambda = max_rho*model.memory/np.array(model.serv_times)/np.array(model.mem_demands)/n
        min_lambda = 0.005/np.array(model.serv_times)/n

        X = np.zeros((samples, n))
        for i in range(n):
            X[:,i] = rng.uniform(min_lambda[i], max_lambda[i], samples)
        return X

    rng = np.random.default_rng()
    m = model.random_model(rng)
    X = create_random_X(rng, m, 10, 0.8)

    k = ApproximateKaufman(m, X)
    print(k.predict(X))

    k2 = ExactKaufman(m)
    print(k2.predict_utility(X))


