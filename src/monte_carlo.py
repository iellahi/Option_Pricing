import numpy as np
from black_scholes import black_scholes_call

# Monte Carlo simulation for European Call Option pricing
def monte_carlo_call(S0, K, T, r, sigma, num_simulations, antithetic=False):
    """
    S0: Current stock price
    K: Strike price of the option
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    num_simulations: Number of simulated terminal prices
    antithetic: If True, use antithetic variates to reduce variance
    Returns (call_price, std_error, ci_lower, ci_upper).
    """
    np.random.seed(42) # For reproducibility
    if antithetic:
        # Draw half as many normals, then pair each draw Z with its negative -Z.
        # The two paths are negatively correlated but both are valid draws from
        # the same distribution, so averaging each pair cancels sampling noise.
        half = num_simulations // 2
        Z = np.random.normal(0, 1, half)
        ST_up = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        ST_down = S0 * np.exp((r - 0.5 * sigma ** 2) * T - sigma * np.sqrt(T) * Z)
        # Average each payoff with its antithetic partner. The pair averages are
        # independent of each other, so the usual standard error formula applies
        # to them (treating the correlated halves as independent would be wrong).
        payoffs = 0.5 * (np.maximum(ST_up - K, 0) + np.maximum(ST_down - K, 0))
        num_draws = half
    else:
        Z = np.random.normal(0, 1, num_simulations) # Standard normal random variables
        # Simulate terminal asset prices at time T using Geometric Brownian Motion
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        # Compute the payoff for each simulation
        payoffs = np.maximum(ST - K, 0)
        num_draws = num_simulations
    # Discount the payoffs to present value
    discounted_payoffs = np.exp(-r * T) * payoffs
    call_price = discounted_payoffs.mean()
    # Standard error of the discounted payoff mean
    std_error = discounted_payoffs.std(ddof=1) / np.sqrt(num_draws)
    # 95% confidence interval
    ci_lower = call_price - 1.96 * std_error
    ci_upper = call_price + 1.96 * std_error
    return call_price, std_error, ci_lower, ci_upper

# Simulate full Geometric Brownian Motion paths for plotting
def simulate_gbm_paths(S0, T, r, sigma, num_paths, num_steps):
    """
    S0: Current stock price
    T: Time to expiration (in years)
    r: Risk-free interest rate (annual rate)
    sigma: Volatility of the stock (annualized)
    num_paths: Number of simulated paths
    num_steps: Number of time steps per path
    Returns an array of shape (num_paths, num_steps + 1).
    """
    np.random.seed(42) # For reproducibility
    dt = T / num_steps
    # Draw all the increments at once
    Z = np.random.normal(0, 1, (num_paths, num_steps))
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    # Build each path step by step using the GBM update rule
    for t in range(1, num_steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )
    return paths

# Example usage:
if __name__ == "__main__":
    S0 = 100 # Current stock price
    K = 95 # Strike price
    T = 1 # Time to expiration (1 year)
    r = 0.05 # Risk-free interest rate (5%)
    sigma = 0.2 # Volatility (20%)
    num_simulations = 100_000 # Number of simulations

    # Standard Monte Carlo estimate
    price, se, ci_lo, ci_hi = monte_carlo_call(S0, K, T, r, sigma, num_simulations)
    print(f"Monte Carlo Call Price (standard): ${price:.4f} (95% CI: ${ci_lo:.4f} to ${ci_hi:.4f})")

    # Antithetic variates estimate
    price_a, se_a, ci_lo_a, ci_hi_a = monte_carlo_call(S0, K, T, r, sigma, num_simulations, antithetic=True)
    print(f"Monte Carlo Call Price (antithetic): ${price_a:.4f} (95% CI: ${ci_lo_a:.4f} to ${ci_hi_a:.4f})")

    # Black-Scholes price for comparison
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    print(f"Black-Scholes Call Price: ${bs_price:.4f}")

    print(f"Standard error (standard): {se:.5f}")
    print(f"Standard error (antithetic): {se_a:.5f}")
