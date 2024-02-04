.. _instruments-toc-doc:

.. ipython:: python
   :suppress:

   from rateslib import *

************
Instruments
************

:ref:`Instruments<instruments-doc>` in *rateslib* are generally categorised into the following
groups:

- **Securities**, which are single currency based, like *Bonds* and *Bills*.
- **Single Currency Derivatives**, like *Interest Rate Swaps (IRS)*, *FRAs*.
- **Multi-currency Derivatives**, like *FXSwaps* and *Cross-Currency Swaps (XCSs)*

Each *Instrument* is its own Python *Class*, and it is sequentially constructed from other classes.

- First :ref:`Periods<periods-doc>` are defined in the ``rateslib.periods`` module.
- Secondly :ref:`Legs<legs-doc>` are defined in the ``rateslib.legs`` module and these
  combine and control a list of organised :ref:`Periods<periods-doc>`.
- Finally :ref:`Instruments<instruments-doc>` are defined in the
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
    d_instruments.rst

The below example demonstrates this composition when creating an :class:`~rateslib.instruments.IRS`.

.. ipython:: python

   irs = IRS(dt(2022, 1, 1), "1Y", "S")
   # The IRS contains 2 Leg attributes.
   irs.leg1
   irs.leg2
   # Each leg contains a list of Periods.
   irs.leg1.periods
   irs.leg2.periods


Examples of Every Instrument
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
