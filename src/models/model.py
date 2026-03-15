# ------------------------------------------------------------------------------
# Autore originale: Prof. Gabriele Russo Russo
# Riadattato da: Brunori Lorenzo per il progetto di tesi magistrale
# Data: Luglio 2025
# ------------------------------------------------------------------------------

import numpy as np

from json import JSONEncoder

class ModelEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Model):
            return o.__dict__
        return o

class Model:
    def __init__ (self, memory, mem_demands, arv_rates, serv_times, serv_cvs, queue_capacity=0, queue_policy="fifo", deadlines=None, utilities=None, init_times=None, map_specs=None, net_overhead = 0):
        self.memory = memory
        self.mem_demands = mem_demands
        self.arv_rates = arv_rates
        self.serv_times = serv_times
        self.queue_capacity = queue_capacity
        self.queue_policy = queue_policy
        self.serv_cvs = serv_cvs
        self.markovian_arrival_processes = map_specs
        self.net_overhead = net_overhead

        if deadlines is None:
            self.deadlines = [float("inf") for f in arv_rates]
        else:
            assert(len(deadlines) == len(serv_times))
            self.deadlines = deadlines

        if utilities is None:
            self.utilities = np.ones(len(self.mem_demands))
        else:
            assert(len(utilities) == len(arv_rates))
            self.utilities = utilities

        if init_times is None:
            self.init_times = np.zeros(len(self.mem_demands))
        else:
            assert(len(init_times) == len(self.mem_demands))
            self.init_times = init_times
        
        if self.markovian_arrival_processes is not None:
            assert(len(self.markovian_arrival_processes) == len(arv_rates))

        assert(len(mem_demands) == len(arv_rates))
        assert(len(serv_times) == len(arv_rates))

    def memoryUtilization (self, lambdas):
        return np.sum(self.memoryUtilization(lambdas))

    def memoryUtilizationPerFlow (self, lambdas):
        util_mat = np.array(self.mem_demands)*np.array(self.serv_times)/self.memory
        return lambdas*util_mat



class Results:

    def __init__(self):
        self.avg_resptime = None
        self.std_resptime = None
        self.blocking_prob = None
        self.blocking_prob_per_fun = None
        self.violation_prob =  None
        self.utility = None
        self.arrivals = None

    def __repr__ (self):
        s = ""
        s += f"totU: {self.utility}\n"
        s += f"pBlock: {self.blocking_prob:.4f}\n"
        s += f"pBlockPerFun: {self.blocking_prob_per_fun}\n"
        s += f"resptime: {self.avg_resptime} (std: {self.std_resptime})\n"
        s += f"pViolation: {self.violation_prob}\n"
        return s


def random_model (rng, n=2, queue_cap=0, arv_rate_multiplier=1, total_mem=2048, deadline_coeff=1.6, queue_policy="fifo", serv_time_exp = False, no_init_times=False, poisson_arrivals=False):
    mem_demands = 128*rng.integers(1,5,size=n)
    arv_rates = rng.uniform(0.5, 5, size=n)*arv_rate_multiplier
    serv_times = rng.uniform(0.1,1, size=n)

    if serv_time_exp:
        serv_cvs = np.ones(n)
    else:
        serv_cvs = rng.uniform(0, 2, size=n)

    if no_init_times:
        init_times = None
    else:
        init_times = rng.uniform(0.05, 0.4, size=n)

    if poisson_arrivals:
        map_specs = None
    else:
        maps=[]
        maps.append("-1.0;1.0") # poisson
        maps.append("-20.0;20.0;0.0;-20.0;0.0;0.0;20.0;0.0") #erlang-2 (SCV 0.5)
        maps.append("-15.0;0.0;0.0;-7.5;7.5;7.5;3.75;3.75") # hyper-exp (SCV 1.23)
        maps.append("-7.6;0.1;3.5;-103.5;7.5;0.0;0.0;100.0") # MMPP(2) (SCV 1.58)

        map_specs = rng.choice(maps, n)

    return Model (total_mem, mem_demands=mem_demands, arv_rates=arv_rates, serv_times=serv_times, serv_cvs=serv_cvs, queue_capacity=queue_cap, queue_policy=queue_policy, deadlines=serv_times*deadline_coeff, init_times=init_times, map_specs=map_specs)


def add_random_functions_to_model(model, rng, n=1, arv_rate_multiplier=1, deadline_coeff=1.6, no_init_times=False, serv_time_exp = True, poisson_arrivals = True):

    # Generiamo randomicamente le caratteristiche delle nuove funzioni
    new_mem_demands = 128*rng.integers(1,5,size=n)
    new_arv_rates = rng.uniform(0.5, 5, size=n)*arv_rate_multiplier
    new_serv_times = rng.uniform(0.1,1, size=n)
    new_deadlines = deadline_coeff * new_serv_times
    if not no_init_times:
        new_init_times = rng.uniform(0.05, 0.4, size=n)
    else: 
        new_init_times = np.zeros(n)

    if serv_time_exp:
        new_serv_cvs = np.ones(n)
    else:
        new_serv_cvs = rng.uniform(0, 2, size=n)

    if poisson_arrivals:
        new_map_specs = None
    else:
        maps=[]
        maps.append("-1.0;1.0") # poisson
        maps.append("-20.0;20.0;0.0;-20.0;0.0;0.0;20.0;0.0") #erlang-2 (SCV 0.5)
        maps.append("-15.0;0.0;0.0;-7.5;7.5;7.5;3.75;3.75") # hyper-exp (SCV 1.23)
        maps.append("-7.6;0.1;3.5;-103.5;7.5;0.0;0.0;100.0") # MMPP(2) (SCV 1.58)
        
        new_map_specs = rng.choice(maps, n)
        model.markovian_arrival_processes = np.concatenate((model.markovian_arrival_processes, new_map_specs))

    # Concateniamo questi nuovi valori a quelli già esistenti
    model.mem_demands = np.concatenate((model.mem_demands, new_mem_demands))
    model.arv_rates = np.concatenate((model.arv_rates, new_arv_rates))
    model.serv_times = np.concatenate((model.serv_times, new_serv_times))
    model.deadlines = np.concatenate((model.deadlines, new_deadlines))
    model.serv_cvs = np.concatenate((model.serv_cvs, new_serv_cvs))
    model.init_times = np.concatenate((model.init_times, new_init_times))

    return model 

def model_from_conf(config):
    functions = config['functions']
    total_mem = config['system_memory']
    mem_demands = np.array(config['mem_demands'])
    arv_rates = np.array(config['arv_rates'])
    serv_times = np.array(config['serv_time_duration'])
    serv_cvs = np.array(config['serv_time_cvs'])
    queue_capacity= config['qlen']
    queue_policy = config['queue_policy']
    deadline_coeff = config['deadline_coeff']
    init_times = np.array(config['init_times'])
    poisson_arrivals = config['poisson_arrivals']
    net_overhead = config['net_overhead']

    if poisson_arrivals:
        map_specs = None
    else:
        raise ValueError("Only Poisson arrivals supported")
    
    if not (len(mem_demands) == len(arv_rates) == len(serv_times) == len(serv_cvs) == len(init_times) == functions):
        raise ValueError("Array dimensions do not match the expected number of functions")


    return Model (memory=total_mem, mem_demands=mem_demands, arv_rates=arv_rates, serv_times=serv_times, serv_cvs=serv_cvs, queue_capacity=queue_capacity, queue_policy=queue_policy, deadlines=serv_times*deadline_coeff, init_times=init_times, map_specs=map_specs, net_overhead=net_overhead)
