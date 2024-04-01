import numpy as np
import matplotlib.pyplot as plt

# Parameters
rate = 5  # Poisson process rate
num_intervals = 1000  # Number of time intervals
num_simulations = 1000  # Number of simulations

# Simulate Poisson process
poisson_events = np.zeros(num_simulations)
for i in range(num_simulations):
    events = 0
    for _ in range(num_intervals):
        uniform_random = np.random.uniform()
        if uniform_random < np.exp(-rate):
            events += 1
    poisson_events[i] = events

# Calculate PDFs
poisson_pdf = np.histogram(poisson_events, bins=np.arange(poisson_events.min(), poisson_events.max()+2), density=True)[0]
exponential_pdf = np.exp(-np.arange(poisson_events.min(), poisson_events.max()+2) * rate)
exponential_pdf = exponential_pdf / np.sum(exponential_pdf)

# Plot PDFs
plt.plot(poisson_pdf, label='Simulated Poisson PDF')
plt.plot(exponential_pdf, label='Exponential PDF with rate = {}'.format(rate))
plt.xlabel('Number of Events')
plt.ylabel('Normalized Probability Density')
plt.legend()
plt.show()