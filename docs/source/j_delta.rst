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

.. warning::

   Setting ``ad`` order manually should never be needed in ``rateslib``.
   The API abstracts and controls this internally. ``ad`` order
   specification is only used in the user guide for examples and to elucidate
   concepts.

.. ipython:: python

   usd_curve = Curve(
       nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97},
       id="sofr_df_",
       ad=1,  # <- this is set automatically when using a Solver.
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
an instantaneous derivative, and it ignores any second order effects so would
only really be accurate for small changes.

Whilst this may, at times, have some use, sensitivity to DFs is not
usually useful. The same application can be made to :class:`~rateslib.curves.LineCurve`
which is instinctively more instructive, and can be compared to the
:meth:`~rateslib.instruments.BaseDerivative.analytic_delta` method (which is scaled
to a basis point (bp) and not 1%)

.. ipython:: python

   usd_curve._set_ad_order(0)
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
:meth:`~rateslib.instruments.Sensitivities.delta`.

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
       id="usd_sofr",
       instruments=instruments,
       s=[2.5, 3.25, 4.0],
       instrument_labels=["1m", "3m", "1y"],
   )
   irs.curves = "sofr"
   irs.delta(solver=usd_solver)

A typical scenario in which FX exposures are created (if the instrument is not
multi-currency) is when ``base`` is set to something other than local currency.

.. ipython:: python

   fxr = FXRates({"eurusd": 1.1})
   irs.fixed_rate = 6.0  # create a negative NPV of approx -11.2k USD
   irs.delta(solver=usd_solver, base="eur", fx=fxr)

The NPV of the :class:`~rateslib.instruments.IRS` in EUR here is approximately -10.2k.
If the EURUSD exchange rate increases by 1000 pips to 1.20, then the EUR NPV increases
to only about -9.3k, meaning a gain of about 900 EUR. The FX sensitivity of about
0.9 EUR/pip is visible in the delta exposure dataframe.