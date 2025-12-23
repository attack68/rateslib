# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

from time import perf_counter

from pandas import DataFrame
from rateslib import IRS, CompositeCurve, Curve, Solver, add_tenor, default_context, dt


class TimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.data = DataFrame(
            {
                "Term": [
                    "1W",
                    "2W",
                    "3W",
                    "1M",
                    "2M",
                    "3M",
                    "4M",
                    "5M",
                    "6M",
                    "7M",
                    "8M",
                    "9M",
                    "10M",
                    "11M",
                    "12M",
                    "18M",
                    "2Y",
                    "3Y",
                    "4Y",
                ],
                "Rate": [
                    5.30111,
                    5.30424,
                    5.30657,
                    5.31100,
                    5.34800,
                    5.38025,
                    5.40915,
                    5.43078,
                    5.44235,
                    5.44950,
                    5.44878,
                    5.44100,
                    5.42730,
                    5.40747,
                    5.3839,
                    5.09195,
                    4.85785,
                    4.51845,
                    4.31705,
                ],
            }
        )
        self.data["Termination"] = [
            add_tenor(dt(2023, 8, 21), _, "F", "nyc") for _ in self.data["Term"]
        ]
        dates = [dt(2023, 8, 17)] + list(self.data["Termination"])
        self.curve = Curve(
            {
                dt(2023, 8, 17): 1.0,  # <- this is today's DF,
                **dict.fromkeys(dates[1:], 0.99),
            }
        )
        self.curve2 = Curve(
            {
                dt(2023, 8, 17): 1.0,  # <- this is today's DF,
                **dict.fromkeys(self.data["Termination"], 1.0),
            },
            t=[dates[0], dates[0], dates[0]] + dates + [dates[-1], dates[-1], dates[-1]],
        )
        self.compcurve = CompositeCurve([self.curve, self.curve2])

    def time_curve_solver(self):
        sofr = Curve(
            id="sofr",
            convention="Act360",
            calendar="nyc",
            modifier="MF",
            interpolation="log_linear",
            nodes={
                dt(2023, 8, 17): 1.0,  # <- this is today's DF,
                **dict.fromkeys(self.data["Termination"], 1.0),
            },
        )
        sofr_args = dict(effective=dt(2023, 8, 21), spec="usd_irs", curves="sofr")
        Solver(
            curves=[sofr],
            instruments=[IRS(termination=_, **sofr_args) for _ in self.data["Termination"]],
            s=self.data["Rate"],
            instrument_labels=self.data["Term"],
            id="us_rates",
        )

    def time_composite_curve_solver_and_values(self):
        sofr = Curve(
            id="sofr",
            convention="Act360",
            calendar="nyc",
            modifier="MF",
            interpolation="log_linear",
            nodes={
                dt(2023, 8, 17): 1.0,  # <- this is today's DF,
                **dict.fromkeys(self.data["Termination"], 1.0),
            },
        )
        spread = Curve(
            id="spread",
            convention="Act360",
            calendar="nyc",
            modifier="MF",
            interpolation="log_linear",
            nodes={
                dt(2023, 8, 17): 1.0,
                dt(2027, 8, 18): 1.0,
            },
        )
        cc = CompositeCurve([sofr, spread], id="us_comp")
        sofr_args = dict(effective=dt(2023, 8, 21), spec="usd_irs", curves="us_comp")
        Solver(
            curves=[sofr, spread, cc],
            instruments=[IRS(termination=_, **sofr_args) for _ in self.data["Termination"]]
            + [IRS(dt(2023, 8, 17), "1b", spec="usd_irs", curves="spread")],
            s=list(self.data["Rate"]) + [0.20],
            instrument_labels=list(self.data["Term"]) + ["spread"],
            id="us_rates",
        )

    def time_curve_value_getting(self):
        dates = [
            dt(2024, 1, 6),
            dt(2024, 2, 6),
            dt(2024, 3, 5),
            dt(2024, 4, 6),
            dt(2024, 5, 6),
            dt(2024, 6, 6),
            dt(2024, 7, 6),
            dt(2024, 8, 6),
            dt(2024, 9, 6),
            dt(2024, 10, 6),
            dt(2024, 11, 6),
        ] * 100
        for _ in dates:
            self.curve[_]

    def time_spline_curve_value_getting(self):
        dates = [
            dt(2024, 1, 6),
            dt(2024, 2, 6),
            dt(2024, 3, 5),
            dt(2024, 4, 6),
            dt(2024, 5, 6),
            dt(2024, 6, 6),
            dt(2024, 7, 6),
            dt(2024, 8, 6),
            dt(2024, 9, 6),
            dt(2024, 10, 6),
            dt(2024, 11, 6),
        ] * 100
        for _ in dates:
            self.curve2[_]

    def time_composite_curve_value_getting(self):
        dates = [
            dt(2024, 1, 6),
            dt(2024, 2, 6),
            dt(2024, 3, 5),
            dt(2024, 4, 6),
            dt(2024, 5, 6),
            dt(2024, 6, 6),
            dt(2024, 7, 6),
            dt(2024, 8, 6),
            dt(2024, 9, 6),
            dt(2024, 10, 6),
            dt(2024, 11, 6),
        ] * 100
        for _ in dates:
            self.compcurve[_]

    def clear_caches(self):
        self.curve._clear_cache()
        self.curve2._clear_cache()

    def time_curve_value_getting_no_cache(self):
        with default_context("curve_caching", False):
            self.time_curve_value_getting()

    def time_spline_curve_value_getting_no_cache(self):
        with default_context("curve_caching", False):
            self.time_spline_curve_value_getting()

    def time_composite_curve_value_getting_no_cache(self):
        with default_context("curve_caching", False):
            self.time_composite_curve_value_getting()


if __name__ == "__main__":
    a = TimeSuite()
    a.setup()
    results = {}
    for bench in [
        # Solvers
        "time_curve_solver",
        "time_composite_curve_solver_and_values",
        # Curve Lookups
        "time_curve_value_getting",
        "time_spline_curve_value_getting",
        "time_composite_curve_value_getting",
        # Curve Lookups No Cache
        "clear_caches",
        "time_curve_value_getting_no_cache",
        "time_spline_curve_value_getting_no_cache",
        "time_composite_curve_value_getting_no_cache",
    ]:
        s_ = perf_counter()
        getattr(a, bench)()
        e_ = perf_counter()
        multiple = int(1 / (e_ - s_))
        s_ = perf_counter()
        for _ in range(multiple):
            getattr(a, bench)()
        e_ = perf_counter()

        results[bench] = (e_ - s_) / multiple

    print("BENCHMARKING RESULTS.......")
    for k, v in results.items():
        print(f"{k}: {v}")
