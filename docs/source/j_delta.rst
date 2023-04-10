.. _delta-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import *
   from datetime import datetime as dt

*****************
Delta Risk
*****************

Fundamental Calculations
------------------------

Sensitivity of :ref:`Instruments<instruments-toc-doc>` can be immediately
extracted from :ref:`Curves<curves-doc>` by initialising a curve with
first (or second order) automatic differentiation (AD) mode.

.. ipython:: python

   usd_curve = Curve(
       nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97},
       id="sofr_df_",
       ad=1
   )
   irs = IRS(
       effective=dt(2022, 1, 1),
       termination="6m",
       frequency="A",
       currency="usd",
   )
   irs.npv(usd_curve)

Observe that from the :class:`~rateslib.dual.Dual` number if the SOFR curve
discount factor at date 1st Jan '23 increases by 1.0, the payer
:class:`~rateslib.instruments.IRS` would lose 511k USD. This is actually
an instantaneous derivative, and that ignores any second order effects so would
only really be accurate for small changes.

Whilst this may, at times, have a some use, sensitivity to DFs is not
usually useful. The same application can be made to :class:`~rateslib.curves.LineCurve`
which is instinctively more instructive, and can be compared to the
:meth:`~rateslib.instruments.BaseDerivative.analytic_delta` method (which is scaled
to a basis point (bp) and not 1%)

.. ipython:: python

   usd_linecurve = LineCurve(
       nodes={dt(2022, 1, 1): 3.0, dt(2023, 1, 1): 4.0},
       id="sofr_on_",
       interpolation="flat_forward",
       ad=1,
   )
   irs.npv([usd_linecurve, usd_curve])
   irs.analytic_delta(usd_linecurve, usd_curve)

Using Solvers
-------------

The :class:`~rateslib.solver.Solver` serves two purposes:

- **solving curves** relative to **calibrating instruments** and market rates,
- obtaining **risk sensitivities** to those **calibrating instruments**.

The mathematical processes involved here are technical and better explained in the
supplementary material (TODO link). Essentially a mapping is created between
the **fundamental calculations** above and the **calibrating instrument rates**.
The :class:`~rateslib.solver.Solver` stores and uses this mapping to create the
:meth:`~rateslib.instruments.Sensitivities.delta`

.. ipython:: python

   usd_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 2, 1): 1.0,
           dt(2022, 4, 1): 1.0,
           dt(2023, 1, 1): 1.0,
        },
       id="sofr",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1m", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "3m", "A", curves="sofr"),
       IRS(dt(2022, 1, 1), "1y", "A", curves="sofr"),
   ]
   usd_solver = Solver(
       curves=[usd_curve],
       id="USD SOFR",
       instruments=instruments,
       s=[2.5, 3.25, 4.0],
       instrument_labels=["1m", "3m", "1y"],
   )
   irs.curves = "sofr"
   irs.delta(solver=usd_solver)