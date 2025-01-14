.. _cook-multi-curves-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Multicurve Framework Construction
******************************************************

The :class:`~rateslib.solver.Solver` is a generalised multi-variable solver targeting the
least squares problem. This means you can generally solve as many *Curves*/*Surfaces* as
necessary, with as many calibration *Instruments* as desired, with the following constraint:
**Solving more variables, via more Instruments, simultaneously, is a harder problem and
costs performance**.

Multi-curve frameworks are common in markets that either have multiple currencies,
or multiple indexes. Examples are the EUR IR market which trades ESTR and Euribor (of
various tenors), NOK, SEK, NZD and AUD, which also trade combinations of RFR indexes
and tenor (IBOR-style) indexes.

This cookbook page will focus on NOK. Why? Because that market contains features which are
useful to discuss and highlight in this article.

.. warning::

   The *Instruments* configured in this article do not precisely match the NOK market, e.g.
   3s6s basis is not a single currency basis swap (*SBS*) but is, in fact, a *Spread* of two *IRSs*.
   Configuring all of this correctly adds unnecessary coding verbosity to a tutorial page.

This cookbook page **will not focus** on: :ref:`adding turns <cook-turns-doc>`, or
using :ref:`different interpolation forms <cook-ptirds-curve-doc>` because these are
covered in other cookbook articles.

There are currently two presented solutions in this article (although other solutions exist):

1) Adding all the instruments, all the curves and all market together into a **single Solver** and letting the optimisation algorithm run.
2) **Identifying independence** of curves, restructuring the instruments and using multiple, sequential solvers to improve performance.

Tradable instruments and data
=============================

In NOK the common tradable instruments, and some of their features, are:

- At the longer end (>3Y) IRSs indexed versus 6M-IBOR (NIBOR).
- At the short end (<3Y) IRSs and FRAs indexed versus 3M-IBOR (NIBOR).
- All products discounted with an RFR (NOWA) curve which has a relatively illiquid market although some ultra short prices are available, and other prices are modelled.
- 3s6s basis market trades across the curve.
- 3sRfr basis market trades across the curve.

We collect the following pricing data on 14th January 2025:

.. ipython:: python

   data = DataFrame({
     "effective": [dt(2025, 1, 16), get_imm(code="h25"),  get_imm(code="m25"),  get_imm(code="u25"),  get_imm(code="z25"),
                   get_imm(code="h26"),  get_imm(code="m26"),  get_imm(code="u26"),  get_imm(code="z26"),
                   get_imm(code="h27"),  get_imm(code="m27"),  get_imm(code="u27"),  get_imm(code="z27")] + [dt(2025, 1, 16)] * 12,
     "termination": [None] + ["3m"] * 12 + ["4y", "5y", "6y", "7y", "8y", "9y", "10y", "12y", "15y", "20y", "25y", "30y"],
     "RFR": [4.50] + [None] * 24,
     "3m": [4.62, 4.45, 4.30, 4.19, 4.13, 4.07, 4.02, 3.98, 3.97, 3.91, 3.88, 3.855, 3.855, None, None, None, None, None, None, None, None, None, None, None, None],
     "6m": [4.62, None, None, None, None, None, None, None, None, None, None, None, None, 4.27, 4.23, 4.20, 4.19, 4.18, 4.17, 4.17, 4.14, 4.07, 3.94, 3.80, 3.66],
     "3s6s Basis": [None, 10.4, 10.4, 10.4, 10.4, 10.4, 10.4, 10.5, 10.5, 10.6, 10.6, 10.5, 10.5, 11.0, 10.9, 11.0, 11.2, 11.6, 12.1, 12.5, 13.8, 15, 16.3, 17.3, 17.8],
     "3sRfr Basis": [None] + [15.5] * 24,
   })
   data

Single global solver
====================

The first thing that is possible is to structure and configure all of these instruments
and data and insert them into a single global solver. There are 75 prices here; 25 for
each curve. First we will construct the curves with node dates in fully consistent and
recognisable positions, defined by the maturity dates of the general instruments.

.. ipython:: python

   termination_dates = [add_tenor(row.effective, row.termination, "MF", "osl") for row in data.iloc[1:].itertuples()]
   data["termination_dates"] = [None] + termination_dates

   # BUILD the Curves
   nowa = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act365f", id="nowa", calendar="osl")
   nibor3 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360", id="nibor3", calendar="osl")
   nibor6 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360", id="nibor6", calendar="osl")

Deposit instruments
-------------------

Let's build the deposit instruments:

.. ipython:: python

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

Outright instruments
--------------------

Next we will build the 3m FRAs and the 6m swaps:

.. ipython:: python

   # Instruments
   ib3_fra = [FRA(row.effective, row.termination, spec="nok_fra3", curves=["nibor3", "nowa"]) for row in data.iloc[1:13].itertuples()]
   ib6_irs = [IRS(row.effective, row.termination, spec="nok_irs6", curves=["nibor6", "nowa"]) for row in data.iloc[13:].itertuples()]

   # Prices
   ib3_fra_s = [_ for _  in data.loc[1:12, "3m"]]
   ib6_irs_s = [_ for _ in data.loc[13:, "6m"]]

   # Labels
   ib3_fra_lbl = [f"fra_{i}" for i in range(1, 13)]
   ib6_irs_lbl = [f"irs_{i}" for i in range(1, 13)]

Basis instruments
-----------------

Now we add the 3s6s basis instruments as single currency basis swaps:

.. ipython:: python

   sbs_irs = [SBS(row.effective, row.termination, spec="nok_sbs36", curves=["nibor3", "nowa", "nibor6", "nowa"]) for row in data.iloc[1:].itertuples()]
   sbs_irs_s = [_ for _ in data.loc[1:, "3s6s Basis"]]
   sbs_irs_lbl =  [f"sbs_{i}" for i in range(1, 25)]

And finally we add the 3sRfr basis instruments. There is not a default specification configured for
this so we define our own.

.. ipython:: python

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
   sbs_rfr_lbl =  [f"sbs_rfr_{i}" for i in range(1, 25)]

Configuring the Solver
----------------------

We add all of the constructions into the *Solver*, and depending on the processor speed of the
machine this might solve in 1-2 seconds.

.. ipython:: python

   solver = Solver(
     curves=[nibor3, nibor6, nowa],
     instruments=rfr_depo + ib3_depo + ib6_depo + ib3_fra + ib6_irs + sbs_irs + sbs_rfr,
     s = rfr_depo_s + ib3_depo_s + ib6_depo_s + ib3_fra_s + ib6_irs_s + sbs_irs_s + sbs_rfr_s,
     instrument_labels = rfr_depo_lbl + ib3_depo_lbl + ib6_depo_lbl + ib3_fra_lbl + ib6_irs_lbl + sbs_irs_lbl + sbs_rfr_lbl,
   )

   nibor3.plot("3m", comparators=[nibor6, nowa], labels=["nibor3", "nibor6", "nowa"])

.. plot:: plot_py/multi_curve_framework.py

   Plotted 3m rates of each curve, NOWA, 3m-NIBOR and 6m-NIBOR.

Independence and using multiple solvers
=======================================

Lets just reset the *Curves* for this next section.

.. ipython:: python

   nowa = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act365f", id="nowa", calendar="osl")
   nibor3 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360", id="nibor3", calendar="osl")
   nibor6 = Curve(nodes={dt(2025, 1, 14): 1.0, dt(2025, 3, 19): 1.0, **{d: 1.0 for d in data.loc[1:, "termination_dates"]}}, convention="act360", id="nibor6", calendar="osl")

The *"nowa"* *Curve* is a primary curve in this scenario. Sometimes it is possible to
refactor the market data quotes to obtain prices in *Instruments* impacting only one curve.
This example does this in a slightly cavalier manner. In practice more care must be taken
in the *Instruments* definitions to ensure the prices are exactly what is expected.

Here we will first solve the *"nowa"* *Curve* by extending the data table (in a simplistic manner),
by subtracting the basis quotes from the outright quotes to obtain pure RFR rates. We will also
do this to obtain 3m rates from 6m rates and vice-versa.

.. ipython:: python

   data.loc[1:12, "RFR"] = data.loc[1:12, "3m"] - data.loc[1:12, "3sRfr Basis"] / 100.0
   data.loc[13:, "RFR"] = data.loc[13:, "6m"] - data.loc[13:, "3s6s Basis"] / 100.0 - data.loc[13:, "3sRfr Basis"] / 100.0
   data.loc[13:, "3m"] = data.loc[13:, "6m"] - data.loc[13:, "3s6s Basis"] / 100.0
   data.loc[1:12, "6m"] = data.loc[1:12, "3m"] + data.loc[1:12, "3s6s Basis"] / 100.0
   data

Preliminary Solver
------------------

Then we can create a Solver which solves the NOWA curve directly:

.. ipython:: python

   solver1 = Solver(
     curves=[nowa],
     instruments=rfr_depo + [IRS(row.effective, row.termination, spec="nok_irs", curves="nowa") for row in data.iloc[1:].itertuples()],
     s = rfr_depo_s + [row.RFR for row in data.iloc[1:].itertuples()],
   )

Additional Solvers in dependency chain
--------------------------------------

This *Curve* is now available to use to price the remaining *Curves*.
We will do the same trick for the rates on the 3M curve.
Notice that we use the ``pre_solvers`` input to pass the already solved *Curve* into the system.

.. ipython:: python

   solver2 = Solver(
     pre_solvers=[solver1],
     curves=[nibor3],
     instruments=ib3_depo + ib3_fra + [IRS(row.effective, row.termination, spec="nok_irs3", curves=["nibor3", "nowa"]) for row in data.iloc[13:].itertuples()],
     s = ib3_depo_s + ib3_fra_s + [row._4 for row in data.iloc[13:].itertuples()],
   )

And finally we repeat this for the 6M Nibor curve. Notice that the total time to solve is about 50%
of the time taken by the single solver system.

.. ipython:: python

   solver3 = Solver(
     pre_solvers=[solver2],
     curves=[nibor6],
     instruments=ib6_depo + [IRS(row.effective, row.termination, spec="nok_irs6", curves=["nibor6", "nowa"]) for row in data.iloc[1:13].itertuples()] + ib6_irs,
     s = ib6_depo_s + [row._5 for row in data.iloc[1:13].itertuples()] + ib6_irs_s,
   )

We can plot the curves.

.. ipython:: python

   nibor3.plot("3m", comparators=[nibor6, nowa], labels=["nibor3", "nibor6", "nowa"])

.. plot:: plot_py/multi_curve_framework2.py

   Plotted 3m rates of each curve, NOWA, 3m-NIBOR and 6m-NIBOR.

The curves look different, and erroneous. This is not because of the method used:
i.e. the alternative framework and
independent solvers. It is because the operation of adding the basis to imply rates of
alternative *Instruments* **is not exact**, in this case, and was too simplistic.
A 3s6s *SBS* rate of 3m-NIBOR + 10bps does not equate to subtracting
10bps from the fixed rate of a 6m *IRS* to yield a 3m *IRS* fixed rate, because the
frequencies ("A", "S" and "Q")
do not align and the conventions ("30e360" and "Act360") do not align either. This is also
a problem for the RFR basis which has a convention of "Act365F" whilst the IBOR type is
"Act360". This approximation has created kinks about the part of the curve where real prices
cross-over to approximated ones.
