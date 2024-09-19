# black_scholes.py

import numpy as np
from scipy import stats
from numpy import log, exp, sqrt

def call_option_price(S, E, T, rf, sigma):
    """
    Calculate the Black-Scholes price for a European call option.

    Parameters:
    S (float): Underlying stock price at time t=0.
    E (float): Strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying stock.

    Returns:
    float: Price of the call option.
    """
    d1 = (log(S / E) + (rf + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print(f"The d1 and d2 parameters for call option: {d1}, {d2}")
    call_price = S * stats.norm.cdf(d1) - E * exp(-rf * T) * stats.norm.cdf(d2)
    return call_price

def put_option_price(S, E, T, rf, sigma):
    """
    Calculate the Black-Scholes price for a European put option.

    Parameters:
    S (float): Underlying stock price at time t=0.
    E (float): Strike price of the option.
    T (float): Time to expiration in years.
    rf (float): Risk-free interest rate.
    sigma (float): Volatility of the underlying stock.

    Returns:
    float: Price of the put option.
    """
    d1 = (log(S / E) + (rf + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print(f"The d1 and d2 parameters for put option: {d1}, {d2}")
    put_price = E * exp(-rf * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
    return put_price

class OptionPricing:
    """
    Class for option pricing using Monte Carlo simulation.
    """

    def __init__(self, S0, E, T, rf, sigma, iterations):
        """
        Initialize the OptionPricing class.

        Parameters:
        S0 (float): Underlying stock price at time t=0.
        E (float): Strike price of the option.
        T (float): Time to expiration in years.
        rf (float): Risk-free interest rate.
        sigma (float): Volatility of the underlying stock.
        iterations (int): Number of Monte Carlo iterations.
        """
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iterations = iterations

    def call_option_simulation(self):
        """
        Calculate the call option price using Monte Carlo simulation.

        Returns:
        float: Simulated price of the call option.
        """
        # Generate random numbers for simulation
        rand = np.random.normal(0, 1, self.iterations)

        # Simulate end-of-period stock prices
        stock_price = self.S0 * np.exp((self.rf - 0.5 * self.sigma ** 2) * self.T
                                       + self.sigma * sqrt(self.T) * rand)

        # Calculate payoffs for the call option
        payoffs = np.maximum(stock_price - self.E, 0)

        # Calculate the present value of the expected payoff
        call_price = exp(-self.rf * self.T) * np.mean(payoffs)
        return call_price

    def put_option_simulation(self):
        """
        Calculate the put option price using Monte Carlo simulation.

        Returns:
        float: Simulated price of the put option.
        """
        # Generate random numbers for simulation
        rand = np.random.normal(0, 1, self.iterations)

        # Simulate end-of-period stock prices
        stock_price = self.S0 * np.exp((self.rf - 0.5 * self.sigma ** 2) * self.T
                                       + self.sigma * sqrt(self.T) * rand)

        # Calculate payoffs for the put option
        payoffs = np.maximum(self.E - stock_price, 0)

        # Calculate the present value of the expected payoff
        put_price = exp(-self.rf * self.T) * np.mean(payoffs)
        return put_price

def main():
    # Parameters
    S0 = 100         # Underlying stock price at t=0
    E = 100          # Strike price
    T = 1            # Time to expiration in years
    rf = 0.05        # Risk-free interest rate
    sigma = 0.2      # Volatility of the underlying stock
    iterations = 1000000  # Number of Monte Carlo iterations

    # Analytical Black-Scholes prices
    print("Analytical Black-Scholes Formula:")
    call_price = call_option_price(S0, E, T, rf, sigma)
    print(f"Call option price according to Black-Scholes model: {call_price}")
    put_price = put_option_price(S0, E, T, rf, sigma)
    print(f"Put option price according to Black-Scholes model: {put_price}")

    # Monte Carlo simulation prices
    print("\nMonte Carlo Simulation:")
    option_pricing = OptionPricing(S0, E, T, rf, sigma, iterations)
    call_price_mc = option_pricing.call_option_simulation()
    print(f"Call option price with Monte Carlo approach: {call_price_mc}")
    put_price_mc = option_pricing.put_option_simulation()
    print(f"Put option price with Monte Carlo approach: {put_price_mc}")

if __name__ == '__main__':
    main()
