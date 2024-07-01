import numpy as np
import matplotlib.pyplot as plt

def algorithm(rate, mu, n):
    """
    Algorithm that computes interevent times and Hawkes intensity for a self-exciting process

    ## Inputs:
    rate: Previous rate
    mu: Background intensity
    n: Weight of the Hawkes process

    ## Outputs: rate x_k, x_k
    x_k: Inter-event time
    rate_tk: Intensity at time tk
    """                                    
    # 1st step
    u1 = np.random.uniform()
    if mu == 0:
        F1 = np.inf
    else:
        F1 = -np.log(u1) / mu

    # 2nd step
    u2 = np.random.uniform()
    if (rate - mu) == 0:
        G2 = 0
    else:
        G2 = 1 + np.log(u2) / (rate - mu)
        

    # 3rd step
    if G2 <= 0:
        F2 = np.inf
    else:
        F2 = -np.log(G2)

    # 4th step
    xk = min(F1, F2)

    # 5th step
    rate_tk = (rate - mu) * np.exp(-xk) + n + mu
    return rate_tk, xk 

def generate_series(K, n, mu):
    """
    Generates temporal series for K Hawkes processes
    
    ##Inputs:
    K: Number of events
    n: Strength of the Hawkes process
    mu: Background intensity 

    ##Output:
    times: time series the events
    rate: time series for the intensity
    """
    times_between_events = [0]
    rate = [mu]
    for _ in range(K):
        rate_tk, xk = algorithm(rate[-1], mu, n)
        rate.append(rate_tk)
        times_between_events.append(xk)
    times = np.cumsum(times_between_events)
    return times, rate

def generate_series_perc(K, n, mu):
    """
    Generates temporal series for K Hawkes processes
    
    ##Inputs:
    K: Number of events
    n: Strength of the Hawkes process
    mu: Background intensity 

    ##Outputs:
    times_between_events: time series the interevent times
    times: time series the events
    rate: time series for the intensity
    """
    times_between_events = [0]
    rate = [mu]
    for _ in range(K):
        rate_tk, xk = algorithm(rate[-1], mu, n)
        rate.append(rate_tk)
        times_between_events.append(xk)
    times = np.cumsum(times_between_events)
    return times_between_events, times, rate

def calculate_percolation_strength(times_between_events, deltas):
    """
    Calculate the percolation strength for a given set of deltas (resolution parameters)

    ## Inputs:
    times_between_events: time series of interevent times
    deltas: list of resolution parameters

    ## Output:
    percolation_strengths: list of percolation strengths
    """

    percolation_strengths = []
    
    for delta in deltas:
        cluster_sizes = []
        current_cluster_size = 1 # The first event is always a cluster
        
        for time in times_between_events:
            if time < delta:
                current_cluster_size += 1
            else:
                if current_cluster_size > 1: # Only consider clusters with more than one event
                    cluster_sizes.append(current_cluster_size)
                current_cluster_size = 1 # The next event is always a cluster
        
        if current_cluster_size > 1: # Consider the last cluster if it ends at the last event
            cluster_sizes.append(current_cluster_size)
         
        if len(cluster_sizes) != 0:  # Check if cluster_sizes is not empty to avoid errors
            max_cluster_size = max(cluster_sizes)
        else:
            max_cluster_size = 0
        
        percolation_strengths.append(max_cluster_size / len(times_between_events))
    
    return percolation_strengths

def identify_clusters(times, delta):
    """
    Identifies clusters in a temporal series given a resolution parameter delta
    
    ## Inputs:
    times: temporal series
    delta: resolution parameter

    ## Output:
    clusters: list of clusters
    """
    clusters = []
    current_cluster = []
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] <= delta:
            if not current_cluster:
                current_cluster.append(times[i])
            current_cluster.append(times[i + 1])
        else:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
    return clusters

def model(n_max, mu_E, mu_I, tau, n_EE, n_IE, n_EI, n_II, dt):
    """
    Solve the equations of the mena field model for a given number of iterations n_max
    
    Inputs:
    n_max: number of iterations
    mu_E: Poisson rate of excitatory neurons
    mu_I: Poisson rate of inhibitory neurons
    tau: characteristic time of the system
    n_EE: influence of excitatory neurons on excitatory neurons
    n_IE: influence of excitatory neurons on inhibitory neurons
    n_EI: influence of inhibitory neurons on excitatory neurons
    n_II: influence of inhibitory neurons on inhibitory neurons
    dt: time step

    Outputs:
    time: time series
    t_events_E: times of events of excitatory neurons
    t_events_I: times of events of inhibitory neurons
    rates_E: rates of excitatory neurons
    rates_I: rates of inhibitory neurons
    """
    n_E = n_I = n = 0
    t_events_E = [0]
    t_events_I = [0]
    rates_E = [mu_E]
    rates_I = [mu_I]
    time = [0]
    while n <= n_max:
        # Excitation neurons
        l_Enew = rates_E[-1]  + dt * (mu_E- rates_E[-1])/tau
        if np.random.uniform() < rates_E[-1]*dt:
            l_Enew += n_EE
            t_events_E.append(time[-1]+dt*np.random.uniform())
            n_E += 1
        if np.random.uniform() < rates_I[-1]*dt:
            l_Enew -= n_IE
            t_events_E.append(time[-1]+dt*np.random.uniform())
            n_E += 1

        # Inhibition neurons
        l_Inew = rates_I[-1] + dt * (mu_I- rates_I[-1])/tau
        if np.random.uniform() < rates_E[-1]*dt:
            l_Inew += n_EI
            t_events_I.append(time[-1]+dt*np.random.uniform())
            n_I += 1
        if np.random.uniform() < rates_I[-1]*dt:
            l_Inew -= n_II
            t_events_I.append(time[-1]+dt*np.random.uniform())
            n_I += 1
        rates_E.append(l_Enew)
        rates_I.append(l_Inew)
        time.append(time[-1]+dt)

        n = n_E + n_I
    return time, t_events_E, t_events_I, rates_E, rates_I

def identify_clusters_model(times, delta):
    """
    Identifies clusters in a temporal series given a resolution parameter delta
    Computes the size and duration of clusters
    
    ## Inputs:
    times: temporal series
    delta: resolution parameter

    ## Output:
    clusters: list of clusters
    clusters_sizes: list of sizes of clusters
    clusters_times: list of durations of clusters
    """
    clusters = []
    current_cluster = []
    for i in range(len(times) - 1):
        if times[i + 1] - times[i] <= delta:
            if not current_cluster:
                current_cluster.append(times[i])
            current_cluster.append(times[i + 1])
        else:
            if current_cluster:
                clusters.append(current_cluster)
                current_cluster = []
    
    clusters_sizes = [len(cluster) for cluster in clusters]
    clusters_times = [cluster[-1] - cluster[0] for cluster in clusters]
    return clusters, clusters_sizes, clusters_times

def bivariate_algorithm(rate1, rate2, muE, muI, nEE, nII, nEI, nIE):
    """
    Algorithm that computes interevent times and Hawkes intensity for a bivariate Hawkes process

    #Inputs:
    rate1: Previous excitation rate
    rate2: Previous inhibition rate
    nEE: "Strength" of the autoexcitation process
    nII: "Strength" of the autoinhibition process
    nEI: "Strength" of the excitation to the inhibition
    nIE: "Strength" of the inhibition to the excitation 
    muE: Background intensity of the excitation
    muI: Background intensity of the inhibition


    #Output: ratex_k, x_k, reaction (0 for excitatory events and 1 for inhibitory events)
    """             
    _, xk1 = algorithm(rate1, muE, nEE)
    _, xk2 = algorithm(rate2, muI, nII)

    xks = [xk1, xk2]

    reaction = np.argmin(xks)

    rate1_tk = 0.
    rate2_tk = 0.

    if reaction == 0:
        rate1_tk = (rate1 - muE) * np.exp(-xk1) + nEE + muE
        rate2_tk = (rate2 - muI) * np.exp(-xk1) + nEI + muI
    else:
        rate1_tk = (rate1 - muE) * np.exp(-xk2) + nIE + muE
        rate2_tk = (rate2 - muI) * np.exp(-xk2) + nII + muI

    
    if rate1_tk <= muE:
        rate1_tk = muE
    if rate2_tk <= muI:
        rate2_tk = muI
        
    xk = xks[reaction]
    
    return rate1_tk, rate2_tk, xk, reaction

def generate_series_bivariate(K, nEE, nII, nEI, nIE, muE, muI):
    """
    Generates temporal series for K bivariate Hawkes processes
    
    ##Inputs:
    K: Number of events
    nEE: "Strength" of the autoexcitation process
    nII: "Strength" of the autoinhibition process
    nEI: "Strength" of the excitation to the inhibition
    nIE: "Strength" of the inhibition to the excitation 
    muE: Background intensity of the excitation
    muI: Background intensity of the inhibition

    ##Output:
    times_between_events: time series the interevent times
    times: time series the events
    rate1: time series for the intensity of process 1 (Excitation)
    rate2: time series for the intensity of process 2 (Inhibition)
    reactions: list the event type (0 for excitation. 1 for inhibition)
    """
    times_between_events = [0]
    rate1 = [muE]
    rate2 = [muI]
    reactions= []
    for _ in range(K):
        rate1_tk, rate2_tk, xk, reaction = bivariate_algorithm(rate1[-1], rate2[-1], muE, muI, nEE, nII, nEI, nIE)
        rate1.append(rate1_tk)
        rate2.append(rate2_tk)
        reactions.append(reaction)
        times_between_events.append(xk)
    times = np.cumsum(times_between_events)
    
    return  times_between_events, times, rate1, rate2, reactions