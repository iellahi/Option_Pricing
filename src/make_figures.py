import os
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import black_scholes_call, black_scholes_put
from monte_carlo import simulate_gbm_paths

# Colors matching the FSU poster palette
GARNET = "#782F40"
TEAL = "#00696B"
GREY = "#555555"

# Reference parameters used throughout
S0 = 100 # Current stock price
K = 95 # Strike price
T = 1 # Time to expiration (1 year)
r = 0.05 # Risk-free interest rate (5%)
sigma = 0.2 # Volatility (20%)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


# Figure 1: simulated GBM stock price paths
def make_gbm_paths_figure():
    num_paths = 30
    num_steps = 252 # trading days in a year
    paths = simulate_gbm_paths(S0, T, r, sigma, num_paths, num_steps)
    time_grid = np.linspace(0, T, num_steps + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    # Plot all paths at low alpha so the highlighted path stands out
    for i in range(num_paths):
        ax.plot(time_grid, paths[i], color=GARNET, alpha=0.25, linewidth=0.8)
    # Highlight one path in full color and heavier weight
    ax.plot(time_grid, paths[0], color=GARNET, alpha=1.0, linewidth=1.8)
    # Horizontal dashed line at the strike price
    ax.axhline(K, color=GREY, linestyle="--", linewidth=1)
    ax.text(T, K, " Strike", color=GREY, va="center", ha="left", fontsize=9)

    ax.set_title("Simulated Stock Price Paths (Geometric Brownian Motion)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Stock Price ($)")
    ax.set_xlim(0, T)

    param_text = f"$S_0$={S0}, $r$={r}, $\\sigma$={sigma}, T={T}"
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", bbox=dict(boxstyle="round", facecolor="white",
                                               edgecolor=GREY, alpha=0.8))

    fig.savefig(os.path.join(FIGURES_DIR, "gbm_paths.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Figure 2: Monte Carlo convergence to the Black-Scholes price
def make_convergence_figure():
    bs_price = black_scholes_call(S0, K, T, r, sigma)

    # Log-spaced sample sizes from 10 to 100,000
    sample_sizes = np.unique(np.logspace(1, 5, 50).astype(int))

    # Draw one large set of standard normals and reuse prefixes of it so the
    # running estimate is a genuine convergence, not independent draws
    np.random.seed(42) # For reproducibility
    Z_full = np.random.normal(0, 1, sample_sizes[-1])

    running_price = np.zeros(len(sample_sizes))
    ci_lower = np.zeros(len(sample_sizes))
    ci_upper = np.zeros(len(sample_sizes))
    running_price_anti = np.zeros(len(sample_sizes))
    ci_lower_anti = np.zeros(len(sample_sizes))
    ci_upper_anti = np.zeros(len(sample_sizes))

    for i, n in enumerate(sample_sizes):
        Z = Z_full[:n]
        ST = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
        running_price[i] = payoffs.mean()
        se = payoffs.std(ddof=1) / np.sqrt(n) if n > 1 else 0
        ci_lower[i] = running_price[i] - 1.96 * se
        ci_upper[i] = running_price[i] + 1.96 * se

        # Antithetic version: pair each Z with -Z within the same prefix and
        # average each pair. The pair averages are independent of each other,
        # so the usual standard error formula applies to them.
        half = max(n // 2, 1)
        Z_half = Z_full[:half]
        ST_up = S0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z_half)
        ST_down = S0 * np.exp((r - 0.5 * sigma ** 2) * T - sigma * np.sqrt(T) * Z_half)
        payoffs_anti = np.exp(-r * T) * 0.5 * (np.maximum(ST_up - K, 0) + np.maximum(ST_down - K, 0))
        running_price_anti[i] = payoffs_anti.mean()
        se_anti = payoffs_anti.std(ddof=1) / np.sqrt(half) if half > 1 else 0
        ci_lower_anti[i] = running_price_anti[i] - 1.96 * se_anti
        ci_upper_anti[i] = running_price_anti[i] + 1.96 * se_anti

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xscale("log")

    # Standard Monte Carlo estimate with shaded 95% CI band
    ax.plot(sample_sizes, running_price, color=GARNET, marker="o", markersize=3,
            linewidth=1, label="Monte Carlo estimate")
    ax.fill_between(sample_sizes, ci_lower, ci_upper, color=GARNET, alpha=0.15)

    # Antithetic estimate, shown dashed with its own tighter band
    ax.plot(sample_sizes, running_price_anti, color=TEAL, linestyle="--", linewidth=1,
            label="Antithetic estimate")
    ax.fill_between(sample_sizes, ci_lower_anti, ci_upper_anti, color=TEAL, alpha=0.15)

    # Reference line at the exact Black-Scholes price
    ax.axhline(bs_price, color=TEAL, linewidth=1.5, label="Black-Scholes Price")

    ax.set_title("Monte Carlo Convergence to the Black-Scholes Price")
    ax.set_xlabel("Number of Simulations")
    ax.set_ylabel("Option Price ($)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    param_text = f"$S_0$={S0}, K={K}, $r$={r}, $\\sigma$={sigma}, T={T}"
    ax.text(0.02, 0.02, param_text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="left", bbox=dict(boxstyle="round", facecolor="white",
                                               edgecolor=GREY, alpha=0.8))

    fig.savefig(os.path.join(FIGURES_DIR, "convergence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Figure 3: option price as a function of volatility
def make_price_vs_volatility_figure():
    sigmas = np.linspace(0.01, 0.50, 200)
    call_prices = [black_scholes_call(S0, K, T, r, s) for s in sigmas]
    put_prices = [black_scholes_put(S0, K, T, r, s) for s in sigmas]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sigmas * 100, call_prices, color=GARNET, linewidth=1.8, label="Call")
    ax.plot(sigmas * 100, put_prices, color=TEAL, linewidth=1.8, label="Put")

    ax.set_title("Option Price and Volatility")
    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Option Price ($)")
    ax.legend(loc="upper left", fontsize=9)

    param_text = f"$S_0$={S0}, K={K}, $r$={r}, T={T}"
    ax.text(0.98, 0.02, param_text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right", bbox=dict(boxstyle="round", facecolor="white",
                                                edgecolor=GREY, alpha=0.8))

    fig.savefig(os.path.join(FIGURES_DIR, "price_vs_volatility.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# Figure 4: option price as a function of the current stock price
def make_price_vs_stock_price_figure():
    stock_prices = np.linspace(50, 125, 200)
    call_prices = [black_scholes_call(s, K, T, r, sigma) for s in stock_prices]
    put_prices = [black_scholes_put(s, K, T, r, sigma) for s in stock_prices]
    # Intrinsic value of the call: max(S - K*exp(-rT), 0)
    intrinsic = np.maximum(stock_prices - K * np.exp(-r * T), 0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(stock_prices, call_prices, color=GARNET, linewidth=1.8, label="Call")
    ax.plot(stock_prices, put_prices, color=TEAL, linewidth=1.8, label="Put")
    ax.plot(stock_prices, intrinsic, color=GREY, linestyle="--", linewidth=1, label="Intrinsic value")
    # Vertical line at the strike price
    ax.axvline(K, color=GREY, linestyle="--", linewidth=1)
    ax.text(K, ax.get_ylim()[1] * 0.95, " Strike", color=GREY, fontsize=9, ha="left", va="top")

    ax.set_title("Option Price and Stock Price")
    ax.set_xlabel("Stock Price ($)")
    ax.set_ylabel("Option Price ($)")
    ax.legend(loc="upper left", fontsize=9)

    param_text = f"K={K}, $r$={r}, $\\sigma$={sigma}, T={T}"
    ax.text(0.98, 0.02, param_text, transform=ax.transAxes, fontsize=8,
            va="bottom", ha="right", bbox=dict(boxstyle="round", facecolor="white",
                                                edgecolor=GREY, alpha=0.8))

    fig.savefig(os.path.join(FIGURES_DIR, "price_vs_stock_price.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    make_gbm_paths_figure()
    make_convergence_figure()
    make_price_vs_volatility_figure()
    make_price_vs_stock_price_figure()
    print("Figures written to figures/")
