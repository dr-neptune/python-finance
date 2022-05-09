from math import log, sqrt, exp
from scipy import stats


def bsm_call_value(S0, K, T, r, sigma):
    """
    Valuation of European call option in Black-Scholes-Merton model

    Args:
        S0: initial stock/index level
        K : strike price
        T : maturity date (in year fractions)
        r : constant risk-free short rate
        sigma: volatility factor in diffusion term

    Returns:
        value: present value of the European call option
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S0 / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    # stats.norm.cdf is the cumulative distribution function for the normal distribution
    value = (S0 * stats.norm.cdf(d1, 0.0, 1.0) -
             K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0))

    return value

def bsm_vega(S0, K, T, r, sigma):
    """
    Vega of European option in BSM model

    Args:
        S0: initial stock / index level
        K : strike price
        T : maturity date (in year fractions)
        r : constant risk-free short rate
        sigma : volatility factor in diffusion term

    Returns:
        vega: partial derivative of BSM formula w.r.t sigma, i.e. vega
    """
    S0 = float(S0)
    d1 = (log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega = S0 * stats.norm.pdf(d1, 0.0, 1.0) * sqrt(T)
    return vega

def bsm_call_imp_vol(S0, K, T, r, C0, sigma_est, it=100):
    """
    Implied volatility of European call option in BSM model

    Args:
        S0: initial stock/index level
        K : strike price
        T : maturity date (in year fractions)
        r : constant risk-free rate
        sigma_est : estimate of impl volatility
        it: number of iterations

    Returns:
        sigma_est: numerically estimated implied volatility
    """
    for i in range(it):
        sigma_est -= ((bsm_call_value(S0, K, T, r, sigma_est) - C0) /
                      bsm_vega(S0, K, T, r, sigma_est))
    return sigma_est
