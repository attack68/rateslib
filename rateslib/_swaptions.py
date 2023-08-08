from math import pi, log, exp
from scipy.stats import norm


def swaption_price(forward, strike, expiry, ann_vol, option, distribution):
    adj_vol = ann_vol * expiry**0.5 / 100
    if distribution == "log":
        x = (log(forward / strike) + 0.5 * adj_vol**2) / adj_vol
        n1, n2 = norm.cdf(x), norm.cdf(x - adj_vol)
        payer_price = forward * n1 - strike * n2
    elif distribution == "normal":
        x = (forward - strike) / adj_vol
        n1 = norm.cdf(x)
        payer_price = (forward - strike) * n1 + (adj_vol / (2 * pi) ** 0.5) * exp(-(x**2) / 2)
    else:
        raise ValueError("`distribution` must be in {'log', 'normal'}.")

    receiver_price = payer_price - (forward - strike)
    if option == "payer":
        return payer_price
    elif option == "receiver":
        return receiver_price
    elif option == "straddle":
        return payer_price + receiver_price
    else:
        raise ValueError("`option` must be in {'receiver', 'payer', 'straddle'}.")


MAXITER = 25
VOLTOL = 1e-4


def swaption_implied_vol(price, forward, strike, expiry, ini_vol, option, distribution):
    v1 = ini_vol
    for i in range(MAXITER):
        v0 = v1
        c0 = swaption_price(forward, strike, expiry, v0, option, distribution)
        c1 = swaption_price(forward, strike, expiry, v0 + 1, option, distribution)
        v1 = v0 + (price - c0) / (c1 - c0)
        if abs(v1 - v0) < VOLTOL:
            return v1
    raise ValueError(f"Failed to converge to tolerance after {MAXITER} iterations.")


def swaption_delta(forward, strike, expiry, ann_vol, option, distribution):
    d0 = swaption_price(forward - 0.0005, strike, expiry, ann_vol, option, distribution)
    d1 = swaption_price(forward + 0.0005, strike, expiry, ann_vol, option, distribution)
    return (d1 - d0) * 1000


def swaption_gamma(forward, strike, expiry, ann_vol, option, distribution):
    g0 = swaption_delta(forward - 0.0005, strike, expiry, ann_vol, option, distribution)
    g1 = swaption_delta(forward + 0.0005, strike, expiry, ann_vol, option, distribution)
    return (g1 - g0) * 10


def swaption_vega(forward, strike, expiry, ann_vol, option, distribution):
    v0 = swaption_price(forward, strike, expiry, ann_vol - 0.0005, option, distribution)
    v1 = swaption_price(forward, strike, expiry, ann_vol + 0.0005, option, distribution)
    return (v1 - v0) * 1000


def swaption_theta(forward, strike, expiry, ann_vol, option, distribution):
    t0 = swaption_price(forward, strike, expiry, ann_vol, option, distribution)
    t1 = swaption_price(forward, strike, expiry - 1 / 252, ann_vol, option, distribution)
    return t1 - t0


def swaption_greeks(*args):
    return [
        swaption_delta(*args),
        swaption_gamma(*args),
        swaption_vega(*args),
        swaption_theta(*args),
    ]
