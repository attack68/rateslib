import matplotlib.pyplot as plt

from rateslib import Curve, Solver, IRS, dt, NoInput

def curve_factory(t):
    return Curve(
        nodes={
            dt(2022, 1, 1): 1.0,
            dt(2022, 3, 15): 1.0,
            dt(2022, 6, 15): 1.0,
            dt(2022, 9, 21): 1.0,
            dt(2022, 12, 21): 1.0,
            dt(2023, 3, 15): 1.0,
            dt(2023, 6, 21): 1.0,
            dt(2023, 9, 20): 1.0,
            dt(2023, 12, 20): 1.0,
            dt(2024, 3, 15): 1.0,
            dt(2025, 1, 1): 1.0,
            dt(2027, 1, 1): 1.0,
            dt(2029, 1, 1): 1.0,
            dt(2032, 1, 1): 1.0,
        },
        convention="act365f",
        calendar="all",
        t=t,
    )


def solver_factory(curve):
    args = dict(calendar="all", frequency="a", convention="act365f", payment_lag=0, curves=curve)
    return Solver(
        curves=[curve],
        instruments=[
            # Deposit
            IRS(dt(2022, 1, 1), "1b", **args),
            # IMMs
            IRS(dt(2022, 3, 15), dt(2022, 6, 15), **args),
            IRS(dt(2022, 6, 15), dt(2022, 9, 21), **args),
            IRS(dt(2022, 9, 21), dt(2022, 12, 21), **args),
            IRS(dt(2022, 12, 21), dt(2023, 3, 15), **args),
            IRS(dt(2023, 3, 15), dt(2023, 6, 21), **args),
            IRS(dt(2023, 6, 21), dt(2023, 9, 20), **args),
            IRS(dt(2023, 9, 20), dt(2023, 12, 20), **args),
            IRS(dt(2023, 12, 20), dt(2024, 3, 15), **args),
            # Swaps
            IRS(dt(2022, 1, 1), "3y", **args),
            IRS(dt(2022, 1, 1), "5y", **args),
            IRS(dt(2022, 1, 1), "7y", **args),
            IRS(dt(2022, 1, 1), "10y", **args)
        ],
        s=[
            # Deposit
            1.0,
            # IMMS
            1.05,
            1.12,
            1.16,
            1.21,
            1.27,
            1.45,
            1.68,
            1.92,
            # Swaps
            1.68,
            2.10,
            2.20,
            2.07
        ]
    )


log_linear_curve = curve_factory(t=NoInput(0))
log_cubic_curve = curve_factory(
    t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 3, 15), dt(2022, 6, 15),
       dt(2022, 9, 21), dt(2022, 12, 21), dt(2023, 3, 15), dt(2023, 6, 21), dt(2023, 9, 20), dt(2023, 12, 20),
       dt(2024, 3, 15), dt(2025, 1, 1), dt(2027, 1, 1), dt(2029, 1, 1), dt(2032, 1, 1), dt(2032, 1, 1), dt(2032, 1, 1),
       dt(2032, 1, 1)])
mixed_curve = curve_factory(
    t=[dt(2024, 3, 15), dt(2024, 3, 15), dt(2024, 3, 15), dt(2024, 3, 15), dt(2025, 1, 1), dt(2027, 1, 1),
       dt(2029, 1, 1), dt(2032, 1, 1), dt(2032, 1, 1), dt(2032, 1, 1), dt(2032, 1, 1)])
solver_factory(log_linear_curve)
solver_factory(log_cubic_curve)
solver_factory(mixed_curve)
fig, ax, line = log_linear_curve.plot("1b", comparators=[log_cubic_curve, mixed_curve],
                                      labels=["log_linear", "log_cubic", "mixed"])
plt.show()
plt.close()