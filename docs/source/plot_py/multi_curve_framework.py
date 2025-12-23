import matplotlib.pyplot as plt

from rateslib import Curve, Solver, IRS, dt, SBS, FRA, get_imm, add_tenor

from pandas import DataFrame

data = DataFrame({
    "effective": [dt(2025, 1, 16), get_imm(code="h25"), get_imm(code="m25"), get_imm(code="u25"),
                  get_imm(code="z25"),
                  get_imm(code="h26"), get_imm(code="m26"), get_imm(code="u26"),
                  get_imm(code="z26"),
                  get_imm(code="h27"), get_imm(code="m27"), get_imm(code="u27"),
                  get_imm(code="z27")] + [dt(2025, 1, 16)] * 12,
    "termination": [None] + ["3m"] * 12 + ["4y", "5y", "6y", "7y", "8y", "9y", "10y", "12y", "15y",
                                           "20y", "25y", "30y"],
    "RFR": [4.50] + [None] * 24,
    "3m": [4.62, 4.45, 4.30, 4.19, 4.13, 4.07, 4.02, 3.98, 3.97, 3.91, 3.88, 3.855, 3.855, None,
           None, None, None, None, None, None, None, None, None, None, None],
    "6m": [4.62, None, None, None, None, None, None, None, None, None, None, None, None, 4.27, 4.23,
           4.20, 4.19, 4.18, 4.17, 4.17, 4.14, 4.07, 3.94, 3.80, 3.66],
    "3s6s Basis": [None, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.5, 10.5, 10.6, 10.6, 10.5, 10.5,
                   11.0, 10.9, 11.0, 11.2, 11.6, 12.1, 12.5, 13.8, 15, 16.3, 17.3, 17.8],
    "3sRfr Basis": [None] + [15.5] * 24,
})

termination_dates = [add_tenor(row.effective, row.termination, "MF", "osl") for row in
                     data.iloc[1:].itertuples()]
data["termination_dates"] = [None] + termination_dates

# BUILD the Curves
nowa = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0,
                    **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act365f",
             id="nowa", calendar="osl")
nibor3 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0,
                      **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360",
               id="nibor3", calendar="osl")
nibor6 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0,
                      **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360",
               id="nibor6", calendar="osl")

# Instruments
rfr_depo = [IRS(dt(2025, 1, 14), "1b", spec="nok_irs", curves="nowa")]
ib3_depo = [IRS(dt(2025, 1, 16), "3m", spec="nok_irs3", curves=["nibor3", "nowa"])]
ib6_depo = [IRS(dt(2025, 1, 16), "6m", spec="nok_irs6", curves=["nibor6", "nowa"])]

# Prices
rfr_depo_s = [data.loc[0, "RFR"]]
ib3_depo_s = [data.loc[0, "3m"]]
ib6_depo_s = [data.loc[0, "6m"]]

# Labels
rfr_depo_lbl = ["rfr_depo"]
ib3_depo_lbl = ["3m_depo"]
ib6_depo_lbl = ["6m_depo"]

# Instruments
ib3_fra = [FRA(row.effective, row.termination, spec="nok_fra3", curves=["nibor3", "nowa"]) for row
           in data.iloc[1:13].itertuples()]
ib6_irs = [IRS(row.effective, row.termination, spec="nok_irs6", curves=["nibor6", "nowa"]) for row
           in data.iloc[13:].itertuples()]

# Prices
ib3_fra_s = [_ for _ in data.loc[1:12, "3m"]]
ib6_irs_s = [_ for _ in data.loc[13:, "6m"]]

# Labels
ib3_fra_lbl = [f"fra_{i}" for i in range(1, 13)]
ib6_irs_lbl = [f"irs_{i}" for i in range(1, 13)]

sbs_irs = [SBS(row.effective, row.termination, spec="nok_sbs36",
               curves=["nibor3", "nowa", "nibor6", "nowa"]) for row in data.iloc[1:].itertuples()]
sbs_irs_s = [_ for _ in data.loc[1:, "3s6s Basis"]]
sbs_irs_lbl = [f"sbs_{i}" for i in range(1, 25)]

args = {
    'frequency': 'q',
    'stub': 'shortfront',
    'eom': False,
    'modifier': 'mf',
    'calendar': 'osl',
    'payment_lag': 0,
    'currency': 'nok',
    'convention': 'act360',
    'leg2_frequency': 'q',
    'leg2_convention': "act365f",
    'spread_compound_method': 'none_simple',
    'fixing_method': "ibor",
    'method_param': 2,
    'leg2_spread_compound_method': 'none_simple',
    'leg2_fixing_method': 'rfr_payment_delay',
    'leg2_method_param': 0,
    'curves': ["nibor3", "nowa", "nowa", "nowa"],
}
sbs_rfr = [SBS(row.effective, row.termination, **args) for row in data.iloc[1:].itertuples()]
sbs_rfr_s = [_ for _ in data.loc[1:, "3sRfr Basis"] * -1.0]
sbs_rfr_lbl = [f"sbs_rfr_{i}" for i in range(1, 25)]

solver = Solver(
    curves=[nibor3, nibor6, nowa],
    instruments=rfr_depo + ib3_depo + ib6_depo + ib3_fra + ib6_irs + sbs_irs + sbs_rfr,
    s=rfr_depo_s + ib3_depo_s + ib6_depo_s + ib3_fra_s + ib6_irs_s + sbs_irs_s + sbs_rfr_s,
    instrument_labels=rfr_depo_lbl + ib3_depo_lbl + ib6_depo_lbl + ib3_fra_lbl + ib6_irs_lbl + sbs_irs_lbl + sbs_rfr_lbl,
)

fig, ax, line = nibor3.plot("3m", comparators=[nibor6, nowa], labels=["nibor3", "nibor6", "nowa"])
plt.show()
plt.close()