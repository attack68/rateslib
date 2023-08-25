
SOFR_FIXED = dict(
    frequency="A",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="nyc",
    payment_lag=2,
    currency="usd",
    convention="Act360",
)

SOFR_FLOAT = dict(
    frequency="A",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="nyc",
    payment_lag=2,
    currency="usd",
    convention="Act360",
    fixing_method="rfr_payment_delay",
    method_param=0,
    spread_compound_method="none_simple",
)

SOFR_FLOAT_AVG = dict(
    frequency="A",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="nyc",
    payment_lag=2,
    currency="usd",
    convention="Act360",
    fixing_method="rfr_payment_delay_avg",
    method_param=0,
    spread_compound_method="none_simple",
)

EU_FIX_CPI_ZERO = dict(
    frequency="Z",
    eom=False,
    modifier="MF",
    calendar="tgt",
    payment_lag=0,
    currency="eur",
    convention="1+",
)

EU_CPI_ZERO = dict(
    frequency="Z",
    eom=False,
    modifier="MF",
    calendar="tgt",
    payment_lag=0,
    currency="eur",
    convention="1+",
    index_lag=3,
    index_method="monthly",
)

EURIBOR_3M = dict(
    frequency="Q",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="tgt",
    payment_lag=0,
    currency="eur",
    convention="Act360",
    fixing_method="ibor",
    method_param=2,
    spread_compound_method="none_simple",
)

EURIBOR_6M = dict(
    frequency="S",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="tgt",
    payment_lag=0,
    currency="eur",
    convention="Act360",
    fixing_method="ibor",
    method_param=2,
    spread_compound_method="none_simple",
)

SONIA_FIXED = dict(
    frequency="A",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="ldn",
    payment_lag=0,
    currency="gbp",
    convention="Act365F",
)

SONIA_FLOAT = dict(
    frequency="A",
    stub="SHORTFRONT",
    eom=False,
    modifier="MF",
    calendar="ldn",
    payment_lag=0,
    currency="gbp",
    convention="Act365F",
    fixing_method="rfr_payment_delay",
    method_param=0,
    spread_compound_method="none_simple",
)

SONIA_FLOAT_ZERO = {**SONIA_FLOAT, **dict(frequency="Z")}

SONIA_FIXED_ZERO = {**SONIA_FIXED, **dict(frequency="Z")}

TEST = dict(
    currency="TES",
    frequency="M",
    leg2_frequency="M",
    convention="TEST",
    leg2_convention="TEST2",
    calendar="nyc,tgt,ldn",
    leg2_calendar="nyc,tgt,ldn",
    modifier="P",
    leg2_modifier="MP",
    stub="LONGFRONT",
    leg2_stub="LONGBACK",
    eom=False,
    leg2_eom=False,
    leg2_roll=1,
    payment_lag=4,
    leg2_payment_lag=3,
)

INSTRUMENT_SPECS = {
    "use_only_for_tests": TEST,
    "us_irs": dict(**SOFR_FIXED, **{"leg2_"+k: v for k, v in SOFR_FLOAT.items()}),
    "us_irs_avg": dict(**SOFR_FIXED, **{"leg2_"+k: v for k, v in SOFR_FLOAT_AVG.items()}),
    "eu_sbs_6m3m": dict(**EURIBOR_6M, **{"leg2_"+k: v for k, v in EURIBOR_3M.items()}),
    "eu_zcis": dict(**EU_FIX_CPI_ZERO, **{"leg2_"+k: v for k, v in EU_CPI_ZERO.items()}),
    "gb_zcs": dict(**SONIA_FIXED_ZERO, **{"leg2_"+k: v for k, v in SONIA_FLOAT_ZERO.items()}),

}