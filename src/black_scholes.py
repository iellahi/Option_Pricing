import math
from scipy.stats import norm

# Black-Scholes formula for European Call Option pricing
def black_scholes_call(S0, K, T, r, sigma):
    """
    S0: Current stock price
    K: Strike price of the option
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    Returns the price of the European call option.
    """
    # Calculate d1 and d2
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Calculate the option price
    call_price = S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    return call_price

# Black-Scholes formula for European Put Option pricing
def black_scholes_put(S0, K, T, r, sigma):
    """
    S0: Current stock price
    K: Strike price of the option
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    Returns the price of the European put option.
    """
    # Calculate d1 and d2
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Put price via the closed form P = K*exp(-rT)*N(-d2) - S0*N(-d1)
    put_price = K * math.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price

# Greeks for the European Call Option
def black_scholes_greeks(S0, K, T, r, sigma):
    """
    S0: Current stock price
    K: Strike price of the option
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    Returns a dict with delta, gamma, vega, theta, rho for the call.
    """
    # Calculate d1 and d2
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    # Delta: sensitivity of the option price to a change in the stock price
    delta = norm.cdf(d1)
    # Gamma: sensitivity of delta to a change in the stock price
    gamma = norm.pdf(d1) / (S0 * sigma * math.sqrt(T))
    # Vega: sensitivity of the option price to a change in volatility (per unit vol)
    vega = S0 * norm.pdf(d1) * math.sqrt(T)
    # Theta: sensitivity of the option price to the passage of time (per year)
    theta = (-(S0 * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2))
    # Rho: sensitivity of the option price to a change in the risk-free rate
    rho = K * T * math.exp(-r * T) * norm.cdf(d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

# Example usage:
if __name__ == "__main__":
    S0 = 100 # Current stock price
    K = 95 # Strike price
    T = 1 # Time to expiration (1 year)
    r = 0.05 # Risk-free interest rate (5%)
    sigma = 0.2 # Volatility (20%)

    # Calculate the call and put option prices
    call_price = black_scholes_call(S0, K, T, r, sigma)
    put_price = black_scholes_put(S0, K, T, r, sigma)
    greeks = black_scholes_greeks(S0, K, T, r, sigma)

    print(f"European Call Option Price: ${call_price:.2f}")
    print(f"European Put Option Price: ${put_price:.2f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")

    # Put-call parity check: C - P should equal S0 - K*exp(-rT)
    parity_lhs = call_price - put_price
    parity_rhs = S0 - K * math.exp(-r * T)
    print(f"Put-Call Parity: C - P = {parity_lhs:.2f}, S0 - K*exp(-rT) = {parity_rhs:.2f}")
