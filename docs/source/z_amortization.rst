.. _cook-amortization-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from rateslib.legs.components import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Applying Amortization to Instruments
******************************************************

Standard Amortization
-----------------------

The :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg` classes both
have ``amortization`` as an input argument which defines the reduction (or addition) of
a notional amount each period.

.. ipython:: python

   curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.65})

.. ipython:: python

   fxl = FixedLeg(
       schedule=Schedule(dt(2000, 1, 1), "1y", "Q"),
       notional=10e6,
       amortization=1e6,      # <- 1mm reduction per period
   )
   fxl.cashflows(disc_curve=curve)[["Type", "Acc Start", "Notional"]]

.. ipython:: python

   fll = FloatLeg(
       schedule=Schedule(dt(2000, 1, 1), "1y", "M"),
       notional=10e6,
       amortization=0.5e6,    # 0.5mm reduction per period
   )
   fll.cashflows(rate_curve=curve)[["Type", "Acc Start", "Notional"]]

*Amortization* is expressed in a specific notional amount reduction per period so,
when applied to an :class:`~rateslib.instruments.IRS`, each leg with different
frequencies should be input directly. Observe the directions.

.. ipython:: python

   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="1Y",
       frequency="Q",
       leg2_frequency="S",
       notional=1e6,
       amortization=2e5,       # <- Reduces notional on 1st July to 600,000
       leg2_amortization=-4e5, # <- Aligns the notional on 1st July
   )
   irs.cashflows(curves=curve)[["Type", "Acc Start", "Notional"]]

Legs with Notional Exchange
----------------------------

If a *Leg* has a *final notional exchange* then any amortised amount would
typically be expected to be paid out at the same time as the notional reduction.
This is visible on :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg`
classes when there is a ``final_exchange``. The final cashflow will be reduced by the
amount of interim exchanges that have already occurred.

.. ipython:: python

   fxl = FixedLeg(
       schedule=Schedule(dt(2000, 1, 1), "1y", "Q"),
       notional=10e6,
       final_exchange=True,
       amortization=1e6,      # <- 1mm reduction and notional exchange per period
   )
   fxl.cashflows(disc_curve=curve)[["Type", "Period", "Acc Start", "Notional"]]

.. ipython:: python

   fll = FloatLeg(
       schedule=Schedule(dt(2000, 1, 1), "1y", "Q"),
       notional=10e6,
       final_exchange=True,
       amortization=1e6,      # <- 1mm reduction and notional exchange per period
   )
   fll.cashflows(rate_curve=curve, disc_curve=curve)[["Type", "Period", "Acc Start", "Notional"]]

An *Instrument* that can potentially use notional exchanges is a *Non-MTM* :class:`~rateslib.instruments.XCS`.

.. ipython:: python

   xcs = XCS(
       effective=dt(2000, 1, 1),
       termination="1y",
       spec="eurusd_xcs",
       notional=5e6,
       amortization=1e6,      # <- 1mm reduction and notional exchange per period
       leg2_mtm=False,
   )
   xcs.cashflows()[["Type", "Period", "Acc Start", "Payment", "Ccy", "Notional"]]


Unsupported
-------------

*Instruments* that currently do **not** support amortization are *Bonds*.

.. ipython:: python

   try:
       FixedRateBond(
           effective=dt(2000, 1, 1),
           termination="1y",
           spec="us_gb",
           notional=5e6,
           amortization=1e6,
           fixed_rate=2.0,
       )
   except Exception as e:
       print(e)

.. ipython:: python

   try:
       IndexFixedRateBond(
           effective=dt(2000, 1, 1),
           termination="1y",
           spec="us_gb",
           notional=5e6,
           amortization=1e6,
           fixed_rate=2.0,
           index_base=100.0,
       )
   except Exception as e:
       print(e)
