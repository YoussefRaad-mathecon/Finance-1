import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Function for the Bachelier model
def Bach(spot, strike, timetomat, t, sigma):
    X = (spot - strike) / (sigma * np.sqrt(timetomat - t))
    prices_bach = (spot - strike) * norm.cdf(X) + sigma * np.sqrt(timetomat - t) * norm.pdf(X)
    return prices_bach

# Function for the standard Black Scholes
def BS(spot, strike, t, r, div, timetomat, sigma):
    d1 = (np.log(spot / strike) + (r - div + 0.5 * sigma ** 2) * (timetomat - t)) / (sigma * np.sqrt(timetomat - t))
    d2 = d1 - sigma * np.sqrt(timetomat - t)
    BS_prices = spot * np.exp(-div * (timetomat - t)) * norm.cdf(d1) - strike * np.exp(-r * (timetomat - t)) * norm.cdf(d2)
    return BS_prices

# Function for the implied volatility
def IV(spot, strike, timetomat, sigma):
    def difference(sigBS):
        return Bach(spot, strike, timetomat, 0, sigma) - BS(spot, strike, 0, 0, 0, timetomat, sigBS)
    return brentq(difference, 1e-6, 10)

# Initialize values
S0 = 100
T = 1
sigma = 15



K = np.arange(50, 150, 0.1)
ImpVol = np.zeros(len(K))

for i, k in enumerate(K):
    ImpVol[i] = IV(S0, k, T, sigma)

# Plotting data
plt.plot(K, ImpVol, color="red", label="Implied Volatilities")
# Highlight point at strike 100
# Plot the blue dot after the line and optionally use the zorder parameter
plt.scatter(100, ImpVol[np.argmin(abs(K-100))], color="blue", zorder=5, label="Strike 100")  
plt.xlabel("Strikes")
plt.ylabel("Implied Volatilities")
plt.title("Implied Volatilities vs. Strikes")
plt.legend()
plt.show()
