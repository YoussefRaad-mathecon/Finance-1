import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

np.random.seed(2024)

rUS = 0.03
rJ = 0
sigmaX = np.array([0.1, 0.02])
sigmaJ = np.array([0, 0.25])
Y0 = 1/100
X0 = Y0
S0 = 30000
K = S0
T = 2

muJ = 0
muX = 0

Nhedge = 252 * 8 * T
Nrep = 10000

def d1(S, K, rJ, sigmaX, sigmaJ, timetomat):
    sigmaJ_norm = np.linalg.norm(sigmaJ)
    return (np.log(S/K) + (rJ - sigmaX.T @ sigmaJ + sigmaJ_norm**2 / 2) * timetomat) / (np.sqrt(timetomat) * sigmaJ_norm)

def d2(S, K, rJ, sigmaX, sigmaJ, timetomat):
    sigmaJ_norm = np.linalg.norm(sigmaJ)
    return (np.log(S/K) + (rJ - sigmaX.T @ sigmaJ - sigmaJ_norm**2 / 2) * timetomat) / (np.sqrt(timetomat) * sigmaJ_norm)

def gdelta(S, Y0, X0, K, rUS, rJ, sigmaX, sigmaJ, timetomat):
    d1_val = d1(S, K, rJ, sigmaX, sigmaJ, timetomat)
    g = Y0 * np.exp((rJ - sigmaX.T @ sigmaJ - rUS) * timetomat) * (norm.cdf(d1_val) - 1)
    deltaQP = g / X0
    return deltaQP

def FQP(S, Y0, K, rUS, rJ, sigmaX, sigmaJ, timetomat):
    d1_val = d1(S, K, rJ, sigmaX, sigmaJ, timetomat)
    d2_val = d2(S, K, rJ, sigmaX, sigmaJ, timetomat)
    priceQP = Y0 * np.exp(-rUS * timetomat) * (K * norm.cdf(-d2_val) - np.exp((rJ - sigmaX.T @ sigmaJ) * timetomat) * S * norm.cdf(-d1_val))
    return priceQP

St = np.full(Nrep, S0)
Xt = np.full(Nrep, X0)
dt = T / Nhedge

initial_outlay = FQP(S0, Y0, K, rUS, rJ, sigmaX, sigmaJ, timetomat=T)
Vpf = np.full(Nrep, initial_outlay)
a = gdelta(St, Y0, Xt, K, rUS, rJ, sigmaX, sigmaJ, timetomat=T)
b = Vpf - a * St * Xt

for i in range(2, Nhedge + 1):
    W = np.random.normal(size=(Nrep, 2))
    St = St * np.exp((muJ + rJ - sigmaX.T @ sigmaJ - 0.5 * np.linalg.norm(sigmaJ)**2) * dt + np.sqrt(dt) * (W @ sigmaJ))
    Xt = Xt * np.exp((muX + rUS - rJ - 0.5 * np.linalg.norm(sigmaX)**2) * dt + np.sqrt(dt) * (W @ sigmaX))
    Vpf = a * St * Xt + b * np.exp(dt * rUS)
    a = gdelta(St, Y0, Xt, K, rUS, rJ, sigmaX, sigmaJ, timetomat=(T - (i - 1) * dt))
    b = (Vpf - a * St * Xt)
import matplotlib.pyplot as plt

Stmax = np.linspace(0, np.max(St), 500)


plt.scatter(St, Vpf, color="blue", s=2, label='Hedge values')
plt.plot(Stmax, Y0 * np.maximum(K - Stmax, 0), 'k-', linewidth=3, label='Payoff') 

plt.xlabel('S(T)')
plt.ylabel('Value of Hedge Portfolio')
plt.title("Value of Hedge Portfolio vs. S(T)",  fontweight='bold')
plt.legend()
ax = plt.gca()  # Get current axes
ax.text(0.15, 0.95, 'Hedge Point Count = 4032', transform=ax.transAxes, color='black', fontweight='bold', verticalalignment='top')
plt.show()
