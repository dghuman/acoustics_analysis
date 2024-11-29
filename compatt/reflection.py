import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mplt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def diff_dist(L):
    D = 94
    d = 24
    a_prime = (L*D)/(d+D)
    a = (d*L)/(d+D)
    c_prime = np.sqrt(D*D + a_prime*a_prime)
    c = np.sqrt(d*d + a*a)
    phi = np.arctan(D/a_prime)
    psi = np.pi - 2*phi
    Lambda = np.sqrt(c_prime*c_prime + c*c - 2*c*c_prime*np.cos(psi))
    return c_prime + c - Lambda

c = 1500

distances = np.linspace(90, 800, 100)
delta_ts = diff_dist(distances)/c
distances_of_interest = np.array([95, 197, 400, 550, 750])
t_of_interest = diff_dist(distances_of_interest)/c

plt.plot(distances, delta_ts*1e3, color=colors[0])
plt.plot(distances_of_interest, t_of_interest*1e3, color=colors[1], linestyle='', marker='x')
for i in range(len(distances_of_interest)):
    plt.annotate(text=f'({distances_of_interest[i]}, {round(t_of_interest[i]*1e3, 2)})', xy=(distances_of_interest[i], t_of_interest[i]*1e3))
plt.title('Reflection Time Difference')
plt.ylabel(r'$\Delta$ Time (ms)')
plt.xlabel('Distance from COMPATT (m)')
plt.show()
