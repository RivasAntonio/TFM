import numpy as np
import matplotlib.pyplot as plt

def poisson_process_discrete(rate, total_time, n_intervals):
    dt = total_time / n_intervals
    probabilities = np.random.uniform(size=n_intervals)
    arrival_times= []
    for i in range(len(probabilities)):
        if probabilities[i] < rate*dt:
            arrival_times.append(i*dt)
    #arrival_times = [i * dt for i in range(n_intervals) if probabilities[i] < rate * dt] # Así también vale, no sé cual es más rápido
    return np.array(arrival_times)

# Parámetros
rate = 0.01  # Tasa de llegada (eventos por unidad de tiempo)
total_time = 10**3  # Tiempo total a simular
n_intervals = total_time*10 # Número de intervalos para la simulación discreta

# Poisson mediante discretización de tiempo
arrival_times_discrete = poisson_process_discrete(rate, total_time, n_intervals)

arrival_times_discrete = arrival_times_discrete-min(arrival_times_discrete)
# Cálculo de los tiempos entre eventos
t_events = np.diff(arrival_times_discrete)

# Tiempos con una distribución exponencial
exponential_times = np.random.exponential(scale=1/rate, size=len(t_events))


# Trama de la distribución exponencial
plt.figure(figsize=(10, 5))
plt.hist(t_events, bins=20, density=True, alpha=0.5, color='blue', label='Intervalos entre eventos')
plt.plot(np.sort(exponential_times), rate * np.exp(-rate * np.sort(exponential_times)), color='r', label='Distribución exponencial')
plt.xlabel('Tiempo entre eventos')
plt.ylabel('Densidad de probabilidad')
plt.title('Comparación')
plt.legend()
plt.tight_layout()
plt.show()

plt.plot(arrival_times_discrete, np.arange(len(arrival_times_discrete)), marker = 'o', drawstyle = 'steps-post')
plt.show()