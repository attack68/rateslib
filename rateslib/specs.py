
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

INSTRUMENT_SPECS = {
    "use_only_for_tests": dict(
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
    ),
    "us_irs": dict(
        **SOFR_FIXED, **{"leg2_"+k: v for k, v in SOFR_FLOAT.items()}
    ),
    "us_irs_avg": dict(
        **SOFR_FIXED, **{"leg2_"+k: v for k, v in SOFR_FLOAT_AVG.items()}
    )
}