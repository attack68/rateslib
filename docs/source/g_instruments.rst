.. _instruments-toc-doc:

.. ipython:: python
   :suppress:

   from rateslib import *

************
Instruments
************

*Instruments* in *rateslib* are generally categorised into the following
groups:

- :ref:`Securities<securities-doc>`, which are single currency based, like *Bonds* and *Bills*.
- :ref:`Single Currency Derivatives<singlecurrency-doc>`, like *Interest Rate Swaps (IRS)*, *FRAs*,
  *Inflation Swaps (ZCISs)*.
- :ref:`Multi-Currency Derivatives<multicurrency-doc>`, like *FXSwaps* and *Cross-Currency Swaps (XCSs)*.
- :ref:`FX Volatility Derivatives<fx-volatility-doc>`, like *FXCalls*, *FXPuts* and *FXStraddles*.
- :ref:`Utilities and Instrument Combinations<combinations-doc>`, which allows things like *Spread trades*,
  *Butterflies*, *Portfolios* and a *Value* for a *Curve*.

Each *Instrument* is its own Python *Class*, and it is sequentially constructed from other classes.

- First :ref:`Periods<periods-doc>` are defined in the ``rateslib.periods`` module.
- Secondly :ref:`Legs<legs-doc>` are defined in the ``rateslib.legs`` module and these
  combine and control a list of organised :ref:`Periods<periods-doc>`.
- Finally *Instruments* are defined in the
  ``rateslib.instruments`` module and these combine and control one or two
  :ref:`Legs<legs-doc>`.

.. image:: _static/p2l2i.png
  :alt: Composition of objects to form instruments
  :width: 433

It is recommended to review the documentation in the above order, since the
composited objects are more explicit in their documentation of each parameter.

Users are expected to rarely use :ref:`Periods<periods-doc>` or
:ref:`Legs<legs-doc>` directly but they are exposed in the public API in order
to construct custom objects.

.. toctree::
    :hidden:
    :maxdepth: 2

    d_periods.rst
    d_legs.rst
    e_securities.rst
    e_singlecurrency.rst
    e_multicurrency.rst
    e_fx_volatility.rst
    e_combinations.rst

The below example demonstrates this composition when creating an :class:`~rateslib.instruments.IRS`.

.. ipython:: python

   irs = IRS(dt(2022, 1, 1), "1Y", "S")
   # The IRS contains 2 Leg attributes.
   irs.leg1
   irs.leg2
   # Each leg contains a list of Periods.
   irs.leg1.periods
   irs.leg2.periods


Examples of Each Instrument
******************************

IRS
----

Suppose *"You Paid $100mm 5Y USD SOFR IRS at 2.5% with normal conventions including
2d payment delay."*

.. ipython:: python

   irs = IRS(
       effective=dt(2023, 12, 20),
       termination="5y",               # effective and termination dates required.
       frequency="a",
       calendar="nyc",
       modifier="mf",
       payment_lag=2,
       roll=20,                        # schedule configuration
       currency="usd",
       notional=100e6,                 # currency and notional
       convention="act360",
       leg2_fixing_method="rfr_payment_delay",
       leg2_method_param=NoInput(0),   # not required
       leg2_fixings=NoInput(0),        # not required
       fixed_rate=2.50,                # pricing parameters
       spec=NoInput(0),                # not required: possible auto-defined IRS exist in defaults.
       curves=NoInput(0),              # not required
    )

**Available** ``spec`` **defaults:**

.. container:: twocol

   .. container:: leftside40

      RFR based swaps:

      - "usd_irs": SOFR swap
      - "eur_irs": ESTR swap
      - "gbp_irs": SONIA swap
      - "sek_irs": SWESTR swap
      - "nok_irs": NOWA swap
      - "chf_irs": SARON swap

   .. container:: rightside60

      IBOR based swaps:

      - "eur_irs3": Euribor 3m swap
      - "eur_irs6": Euribor 6m swap
      - "eur_irs1": Euribor 1m swap
      - "sek_irs3": Stibor 3m swap
      - "nok_irs3": Nibor 3m swap
      - "nok_irs6": Nibor 6m swap

.. raw:: html

   <div class="clear"></div>


FRA
---

Suppose *"You Paid 500m 1x4 28th FRA in EUR at 2.5%".*

.. ipython:: python

   fra = FRA(
       effective=dt(2022, 2, 28),
       termination="3m",
       frequency="q",
       roll=28,
       eom=False,
       modifier="mf",
       calendar="tgt",
       payment_lag=0,
       notional=500e6,
       currency="eur",
       convention="act360",
       method_param=2,
       fixed_rate=2.50,
       fixings=NoInput(0),
       curves=["euribor3m", "estr"],
       spec=NoInput(0),
   )

**Available** ``spec`` **defaults:**

- "eur_fra3": Euribor 3M FRA
- "eur_fra6": Euribor 6M FRA
- "sek_fra3": Stibor 3M FRA

SBS
----

Suppose *"You Paid 5Y EUR single-swap 3s6s basis at 6.5bp in €100mm."*

.. ipython:: python

   sbs = SBS(
       effective=dt(2023, 12, 20),
       termination="5y",
       frequency="q",
       leg2_frequency="s",          # effective, termination and frequency dates required.
       calendar="tgt",
       modifier="mf",
       roll=20,                     # schedule configuration
       currency="eur",
       notional=100e6,              # currency and notional
       convention="act360",
       fixing_method="ibor",
       method_param=2,
       leg2_fixing_method="ibor",
       leg2_method_param=2,
       # fixings=NoInput(0),
       # leg2_fixings=NoInput(0),
       float_spread=6.50            # pricing parameters
       # spec=NoInput(0),           # possible auto-defined IRS exist in defaults.
       # curves=["euribor3m", "estr", "euribor6m", "estr"], # curves optional.
   )

**Available** ``spec`` **defaults:**

- "eur_sbs36": Euribor 3m vs 6m single basis swap

ZCS
----

Suppose *"You Received £100mm 15Y Zero Coupon Swap effective
16th March 2023 at IRR of 2.15% (under Act365F),
compounding annually."*

.. ipython:: python

   zcs = ZCS(
       effective=dt(2023, 3, 16),
       termination="15Y",
       frequency="A",                # effective, termination and compounding frequency required.
       modifier="mf",
       eom=True,
       calendar="ldn",               # schedule configuration
       currency="gbp",
       notional=-100e6,              # currency and notional
       convention="act365f",
       fixed_rate=2.15,
       leg2_fixing_method="rfr_payment_delay",
       leg2_fixings=NoInput(0),
       leg2_method_param=0,          # pricing parameters
       curves="gbpgbp",              # curves for forecasting and discounting each leg.
       spec=NoInput(0),              # possible auto-defined ZCS exist in defaults.
    )

**Available** ``spec`` **defaults:**

- "gbp_zcs": GBP Zero coupon swap

ZCIS
-----

Suppose *"You received €100m 5Y Zero Coupon Inflation Swap at an IRR of 1.15% versus EUR-CPI
with a 3-month lag".*

.. ipython:: python

   zcis = ZCIS(
       effective=dt(2022, 1, 1),
       termination="5Y",
       frequency="A",
       calendar="tgt",
       modifier="mf",
       currency="eur",
       fixed_rate=1.15,
       convention="1+",
       notional=-100e6,
       leg2_index_base=100.0,
       leg2_index_method="monthly",
       leg2_index_lag=3,
       leg2_index_fixings=NoInput(0),
       curves=[None, "estr", "eur_cpi", "estr"],
       spec=NoInput(0),
   )

**Available** ``spec`` **defaults:**

- "eur_zcis": EUR Zero coupon inflation swap
- "gbp_zcis": GBP Zero coupon inflation swap
- "usd_zcis": USD Zero coupon inflation swap

IIRS
-----

Suppose *"You bought an inflation asset swap, by Paying £100mm 5Y IIRS effective
1st Jan 2022 at 1.15% indexed at GBP-CPI with a 3 month lag, versus SONIA paid semi-annually
@ -50bps."*

.. ipython:: python

   iirs = IIRS(
       effective=dt(2022, 1, 1),
       termination="5Y",
       frequency="S",
       calendar="ldn",
       modifier="none",
       leg2_modifier="mf",
       currency="gbp",
       fixed_rate=1.15,
       payment_lag=0,
       convention="ActActICMA",
       leg2_convention="act365f",
       notional=100e6,
       index_base=100.0,
       index_method="monthly",
       index_lag=3,
       index_fixings=NoInput(0),
       leg2_fixing_method="rfr_payment_delay",
       leg2_fixings=NoInput(0),
       leg2_float_spread=-50.0,
       curves=["gbp_cpi", "sonia", "sonia", "sonia"],
       spec=NoInput(0),
   )

**Available** ``spec`` **defaults:**

- "sek_iirs3": SEK inflation IRS versus Stibor 3m

FXExchange
-----------

Suppose *"You Bought £100mm Selling $125mm (GBPUSD 1.25) for settlement 16th March 2023."*

.. ipython:: python

   fxe = FXExchange(
       settlement=dt(2023, 3, 16),
       currency="gbp",
       leg2_currency="usd",
       fx_rate=1.25,
       notional=100e6,
       curves=[None, "gbpusd", None, "usdusd"],  # curves for discounting each currency, optional
   )

FXSwap
-------

Suppose *"You Paid 3M EURUSD FX Swap in €100mm/$105mm (split notional €101.5mm) at +40.2 points."*

.. ipython:: python

   fxs = FXSwap(
       effective=dt(2023, 12, 20),
       termination="3m",         # effective and termination dates required.
       calendar="nyc,tgt",
       modifier="mf",            # calendar and modifier to determine termination.
       currency="eur",
       leg2_currency="usd",      # currencies are specified directly in lowercase.
       notional=-100e6,
       split_notional=-101.5e6,  # split notional defined explicitly.
       fx_fixings=1.05,
       points=40.2,              # all pricing parameters are defined
       spec=NoInput(0),          # not required.
       curves=[None, "eurusd", None, "usdusd"],  # curves for discounting each leg optional.
   )

**Available** ``spec`` **defaults:**

- "eurusd_fxs": EUR/USD FX swap
- "gbpusd_fxs": GBP/USD FX swap

XCS
----

Suppose *"You Paid €100mm 5Y EUR/USD MTM Cross currency swap @-15bp under normal RFR conventions,
with initial FX fixing agreed at EURUSD 1.08."*

.. ipython:: python

   xcs = XCS(
       effective=dt(2023, 12, 20),
       termination="5y",            # effective and termination dates required.
       frequency="q",
       calendar="tgt,nyc",
       modifier="mf",
       payment_lag=2,
       payment_lag_exchange=0,
       roll=20,                     # schedule configuration
       currency="eur",
       leg2_currency="usd",
       fx_fixings=1.08,
       notional=100e6,              # currency and notional
       fixed=False,
       leg2_fixed=False,
       leg2_mtm=True,
       convention="act360",
       fixing_method="rfr_payment_delay",
       leg2_fixing_method="rfr_payment_delay",
       method_param=NoInput(0),
       leg2_method_param=NoInput(0),
       fixings=NoInput(0),
       leg2_fixings=NoInput(0),
       float_spread=-15.0,           # pricing parameters
       spec=NoInput(0),              # possible auto-defined XCS exist in defaults.
       curves=["eureur", "eurusd", "usdusd", "usdusd"],  # curves optional.
   )

**Available** ``spec`` **defaults:**

- "eurusd_xcs": EUR/USD MTM Cross currency swap
- "gbpusd_xcs": GBP/USD MTM Cross currency swap
- "eurgbp_xcs": EUR/GBP MTM Cross currency swap
