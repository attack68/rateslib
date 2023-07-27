from pandas.tseries.offsets import BusinessDay

# Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
# Commercial use of this code, and/or copying and redistribution is prohibited.
# Contact rateslib at gmail.com if this code is observed outside its intended sphere.


class Defaults:
    """
    Test docs
    """

    convention = "ACT360"
    notional = 1.0e6
    stub = "SHORTFRONT"
    stub_length = "SHORT"
    modifier = "MF"
    index_lag = 3
    index_method = "daily"
    payment_lag = 2
    payment_lag_exchange = 0
    payment_lag_specific = {
        "IRS": 2,
        "IIRS": 2,
        "ZCS": 2,
        "ZCIS": 0,
        "FXSwap": 0,
        "SBS": 2,
        "Swap": 2,
        "NonMtmXCS": 2,
        "NonMtmFixedFloatXCS": 2,
        "NonMtmFixedFixedXCS": 2,
        "XCS": 2,
        "FixedFloatXCS": 2,
        "FixedFixedXCS": 2,
        "FloatFixedXCS": 2,
        "FixedRateBond": 0,
        "IndexFixedRateBond": 0,
        "FloatRateBond": 0,
        "Bill": 0,
        "FRA": 0,
    }
    calendar = BusinessDay()
    interpolation = {
        "Curve": "log_linear",
        "LineCurve": "linear",
        "IndexCurve": "linear_index",
    }
    endpoints = "natural"
    frequency_months = {
        "M": 1,
        "B": 2,
        "Q": 3,
        "T": 4,
        "S": 6,
        "A": 12,
        "Z": 1e8,
    }
    eom = False
    fixing_method = "rfr_payment_delay"
    fixing_method_param = {
        "rfr_payment_delay": 0,  # no observation shift - use payment_delay param
        "rfr_observation_shift": 2,
        "rfr_lockout": 2,
        "rfr_lookback": 2,
        "ibor": 2,
    }
    spread_compound_method = "none_simple"
    base_currency = "usd"
    fx_swap_base = "foreign"
    tag = "v"
    headers = {
        "type": "Type",
        "stub_type": "Period",
        "u_acc_start": "Unadj Acc Start",
        "u_acc_end": "Unadj Acc End",
        "a_acc_start": "Acc Start",
        "a_acc_end": "Acc End",
        "payment": "Payment",
        "convention": "Convention",
        "dcf": "DCF",
        "df": "DF",
        "notional": "Notional",
        "currency": "Ccy",
        "rate": "Rate",
        "spread": "Spread",
        "npv": "NPV",
        "cashflow": "Cashflow",
        "fx": "FX Rate",
        "npv_fx": "NPV Ccy",
        "real_cashflow": "Real Cashflow",
        "index_value": "Index Val",
        "index_ratio": "Index Ratio",
        "index_base": "Index Base",
    }
    algorithm = "gauss_newton"
    curve_not_in_solver = "ignore"
    no_fx_fixings_for_xcs = "warn"
    pool = 1

    # fmt: off
    multi_csa_steps = [
       2, 5, 10, 20, 30, 50, 77, 81, 86, 91, 96, 103, 110, 119, 128, 140, 153,
       169, 188, 212, 242, 281, 332, 401, 498, 636, 835, 1104, 1407, 1646, 1766,
       1808, 1821, 1824, 1825,
    ]
    # fmt: on

    # Licence: Creative Commons - Attribution-NonCommercial-NoDerivatives 4.0 International
    # Commercial use of this code, and/or copying and redistribution is prohibited.
    # Contact rateslib at gmail.com if this code is observed outside its intended sphere.

    def reset_defaults(self):
        base = Defaults()
        for attr in [_ for _ in dir(self) if "__" != _[:2]]:
            setattr(self, attr, getattr(base, attr))


def plot(x, y: list, labels=[]):
    import matplotlib.pyplot as plt  # type: ignore[import]
    import matplotlib.dates as mdates  # type: ignore[import]

    fig, ax = plt.subplots(1, 1)
    lines = []
    for _y in y:
        (line,) = ax.plot(x, _y)
        lines.append(line)
    if labels and len(labels) == len(lines):
        ax.legend(lines, labels)
    years = mdates.YearLocator()  # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter("%Y")
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.grid(True)
    fig.autofmt_xdate()
    return fig, ax, lines
