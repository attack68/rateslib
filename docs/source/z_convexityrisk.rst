.. _conv-risk-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Building a Risk Framework Including STIR Convexity Adjustments
****************************************************************

Common risk frameworks that I have experienced, used :class:`~rateslib.instruments.STIRFuture`
prices to construct IR *Curves* at the short end. However they typically did so by implying a
direct curve rate from the future's price, and then adding a convexity adjustment as a parameter,
manually.

When portfolio managers want to know their delta sensitivity versus the convexity parameter itself
they can simply add up the number of *STIRFuture* contracts they have and compare versus swaps. This
is trivial to do, but, it does not provide a *complete* risk sensitivity framework, or
:class:`~rateslib.solver.Solver` - it requires thos extra steps. What this page shows is how to
create a *Solver* for the first 2-years of the
*Curve* including convexity instruments so that risk against :class:`~rateslib.instruments.IRS`
(and/or any other relevant *Instruments*) can be actively calculated.

First we construct a :class:`~rateslib.curves.Curve` that is used to calculate *STIR Future* rates.
In this framework instrument prices are given in *rate* terms (i.e. price = 100 - rate).

.. ipython:: python

   curve_stir = Curve(
       nodes={
           dt(2023, 11, 2): 1.0,
           # dt(2023, 12, 20): 1.0, need only the maturity of each STIRF
           dt(2024, 3, 20): 1.0,
           dt(2024, 6, 19): 1.0,
           dt(2024, 9, 18): 1.0,
           dt(2024, 12, 18): 1.0,
           dt(2025, 3, 19): 1.0,
           dt(2025, 6, 18): 1.0,
           dt(2025, 9, 17): 1.0,
           dt(2025, 12, 17): 1.0,
       },
       calendar="nyc",
       convention="act360",
       id="stir",
   )

Next we define the instruments and construct the risk framework *Solver*.

.. ipython:: python

   instruments_stir = [
       STIRFuture(dt(2023, 12, 20), dt(2024, 3, 20), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2024, 3, 20), dt(2024, 6, 19), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2024, 6, 19), dt(2024, 9, 18), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2024, 9, 18), dt(2024, 12, 18), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2024, 12, 18), dt(2025, 3, 19), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2025, 3, 19), dt(2025, 6, 18), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2025, 6, 18), dt(2025, 9, 17), spec="sofr3mf", curves="stir"),
       STIRFuture(dt(2025, 9, 17), dt(2025, 12, 17), spec="sofr3mf", curves="stir"),
   ]
   stir_solver = Solver(
       curves=[curve_stir],
       instruments=instruments_stir,
       s=[4.0, 3.75, 3.5, 3.25, 3.15, 3.10, 3.08, 3.07],
       instrument_labels=["z23", "h24", "m24", "u24", "z24", "h25", "m25", "u25"],
       id="STIRF"
   )

This *Solver* calculates risk sensitivities against SOFR future rates. It can be used
to risk SOFR futures directly or risk :class:`~rateslib.instruments.IRS` that have been
mapped specifically to use the *"stir"* curve. This is not entirely accurate because *IRS* should be
priced from a convexity adjusted *IRS* curve.

.. ipython:: python

   stirf = STIRFuture(dt(2024, 9, 18), dt(2024, 12, 18), spec="sofr3mf", curves="stir", contracts=1000)
   stirf.delta(solver=stir_solver)

.. ipython:: python

   irs = IRS(dt(2024, 9, 18), dt(2024, 12, 18), spec="usd_irs", curves="irs", notional=1e9)
   irs.delta(curves="stir", solver=stir_solver)

Adding convexity adjustments
------------------------------

Now that we have a *Curve* which defines *STIR Future* prices we can use a
:class:`~rateslib.instruments.Spread` to relate these prices to *IRS* prices and the
*IRS* *Curve* (technically this *Curve* does not have to have the same structure as the
*"stir"* *Curve* but for for purposes of example it inherits it for simplicity's sake).

.. ipython:: python

   curve_irs = Curve(
       nodes={
           dt(2023, 11, 2): 1.0,
           # dt(2023, 12, 20): 1.0, need only the maturty of each STIRF
           dt(2024, 3, 20): 1.0,
           dt(2024, 6, 19): 1.0,
           dt(2024, 9, 18): 1.0,
           dt(2024, 12, 18): 1.0,
           dt(2025, 3, 19): 1.0,
           dt(2025, 6, 18): 1.0,
           dt(2025, 9, 17): 1.0,
           dt(2025, 12, 17): 1.0,
       },
       calendar="nyc",
       convention="act360",
       id="irs",
   )

The *Instruments* are set to be *Spreads* between the original *STIR Futures* and an
*IRS* (or potentially an *FRA*) measured over the same dates.

.. ipython:: python

   instruments_irs = [
       Spread(instruments_stir[0], IRS(dt(2023, 12, 20), dt(2024, 3, 20), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[1], IRS(dt(2024, 3, 20), dt(2024, 6, 19), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[2], IRS(dt(2024, 6, 19), dt(2024, 9, 18), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[3], IRS(dt(2024, 9, 18), dt(2024, 12, 18), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[4], IRS(dt(2024, 12, 18), dt(2025, 3, 19), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[5], IRS(dt(2025, 3, 19), dt(2025, 6, 18), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[6], IRS(dt(2025, 6, 18), dt(2025, 9, 17), spec="usd_irs", curves="irs")),
       Spread(instruments_stir[7], IRS(dt(2025, 9, 17), dt(2025, 12, 17), spec="usd_irs", curves="irs")),
   ]

Finally, we add these into a new dependent *Solver* (we do not have to create a
dependency chain of *Solvers* we could do this all simultaneously in a single *Solver*, but
it is better elucidated this way).

.. ipython:: python

   full_solver = Solver(
       pre_solvers=[stir_solver],
       curves=[curve_irs],
       instruments=instruments_irs,
       s=[0.07, 0.25, 0.5, 0.95, 1.4, 1.8, 2.2, 2.6],
       instrument_labels=["z23", "h24", "m24", "u24", "z24", "h25", "m25", "u25"],
       id="Convexity",
   )

Now we can re-risk the original instruments as part of the extended risk framework.

.. ipython:: python

   stirf.delta(solver=full_solver)

.. ipython:: python

   irs.delta(solver=full_solver)

We can even combine the instruments into a single :class:~rateslib.instruments.Portfolio`
and observe the combined risk analytics.

.. ipython:: python

   pf = Portfolio([stirf, irs])
   pf.delta(solver=full_solver)
