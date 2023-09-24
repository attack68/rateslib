.. RatesLib documentation master file,
.. _guide-doc:

==========
User Guide
==========

Where to start?
===============

It is important to understand that the key elements of this library are
**curve construction**, **financial instrument specification**,
**foreign exchange (FX)** and **risk sensitivity**.
All of these functionalities are interlinked and potentially dependent upon each
other. This guide's intention is to introduce them in a structured way.

Let's start with a basic *Curve* and *Instrument*.

A trivial example
----------------------------

For example, we can construct :ref:`Curves<c-curves-doc>` in many different ways:
here we create one by directly specifying discount factors (DFs) on certain node dates (sometimes
called pillar dates in other publications).

.. ipython:: python

   from rateslib import dt
   from rateslib.curves import Curve
   usd_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 7, 1): 0.98,
           dt(2023, 1, 1): 0.95
       },
       calendar="nyc",
       id="sofr",
   )

We can then construct an :ref:`Instrument<instruments-toc-doc>`. Here we create a short dated
RFR interest rate swap (:class:`~rateslib.instruments.IRS`) using market specification pre-defined
by *rateslib*.

.. ipython:: python

   from rateslib.instruments import IRS
   irs = IRS(
       effective=dt(2022, 2, 15),
       termination="6m",
       notional=1000000000,
       fixed_rate=2.0,
       spec="usd_irs"
   )

You can read more about arguments for instruments :ref:`here<defaults-doc>`. To double check
some of the arguments for a *"usd_irs"* one can run:

.. ipython:: python

   from rateslib import defaults
   defaults.spec["usd_irs"]


We can value the *IRS* with the *Curve* in its local currency (USD) by default, and see
the generated cashflows.

.. ipython:: python

   irs.npv(usd_curve)

.. ipython:: python

   irs.cashflows(usd_curve)

If instead of this trivial, minimalist example you would like to see a real world
example :ref:`replicating a Bloomberg SWPM function SOFR curve<cook-swpm-doc>` please
click the link.

Quick look at FX
==================

Spot rates and conversion
-------------------------

The above values were all calculated and displayed in USD. That is the default
currency in *rateslib* and the local currency of the swap. We can convert this value to another
currency using the :class:`~rateslib.fx.FXRates` class. This is a basic class which is
parametrised by some exchange rates.

.. ipython:: python

   from rateslib.fx import FXRates, FXForwards
   fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.25})
   fxr.rates_table()

We now have a mechanism by which to specify values in other currencies.

.. ipython:: python

   irs.npv(usd_curve, fx=fxr, base="usd")
   irs.npv(usd_curve, fx=fxr, base="eur")

One observes that the value returned here is not a float but a :class:`~rateslib.dual.Dual`
which is part of *rateslib's* AD framework. One can read more about this particular treatment of FX
:ref:`here<fx-dual-doc>` and more generally about the dual AD framework :ref:`here<dual-doc>`.

FX forwards
------------

For multi-currency derivatives we need more than basic, spot exchange rates.
We can also create an
:class:`~rateslib.fx.FXForwards` class. This stores the FX rates and the interest
rates curves that are used for all the FX-interest rate parity derivations. With these
we can calculate forward FX rates and also ad-hoc FX swap rates.

.. ipython:: python

   eur_curve = Curve({
       dt(2022, 1, 1): 1.0,
       dt(2022, 7, 1): 0.972,
       dt(2023, 1, 1): 0.98},
       calendar="tgt",
   )
   eurusd_curve = Curve({
       dt(2022, 1, 1): 1.0,
       dt(2022, 7, 1): 0.973,
       dt(2023, 1, 1): 0.981}
   )
   fxf = FXForwards(
       fx_rates=FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 1)),
       fx_curves={
           "usdusd": usd_curve,
           "eureur": eur_curve,
           "eurusd": eurusd_curve,
       }
   )
   fxf.rate("eurusd", settlement=dt(2023, 1, 1))
   fxf.swap("eurusd", settlements=[dt(2022, 2, 1), dt(2022, 5, 1)])

*FXForwards* objects are comprehensive and more information regarding all of the
:ref:`FX features<fx-doc>` is available in this link.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_fx.rst

More about instruments
======================

We saw an example of the :class:`~rateslib.instruments.IRS` instrument above.
A complete guide for all of the :ref:`Instruments<instruments-toc-doc>` is available in
this link. It is
recommended to also, in advance, review :ref:`Periods<periods-doc>` and
then :ref:`Legs<legs-doc>`, since
the documentation for these building blocks provides technical descriptions of the
parameters that are used to build up the instruments.

Multi-currency instruments
--------------------------

Lets take a quick look at a multi-currency instrument: the
:class:`~rateslib.instruments.FXSwap`. All instruments have a mid-market pricing
function :meth:`rate()<rateslib.instruments.BaseDerivative.rate>` which is used as the
target for the *Solver*.

.. ipython:: python

   from rateslib.instruments import FXSwap
   fxs = FXSwap(
       effective=dt(2022, 2, 1),
       termination="3m",
       notional=20e6,
       currency="eur",
       leg2_currency="usd",
   )
   fxs.rate(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)

Securities and bonds
--------------------

A very common instrument in financial investing is a :class:`~rateslib.instruments.FixedRateBond`.
At time of writing the on-the-run 10Y US treasury was the 3.875% Aug 2033 bond. Here we can
construct this using the street convention and derive the price from yield-to-maturity and
risk calculations.

.. ipython:: python

   from rateslib import FixedRateBond
   fxb = FixedRateBond(
       effective=dt(2023, 8, 15),
       termination=dt(2033, 8, 15),
       fixed_rate=3.875,
       spec="ust"
   )
   fxb.accrued(settlement=dt(2025, 2, 14))
   fxb.price(ytm=4.0, settlement=dt(2025, 2, 14))
   fxb.duration(ytm=4.0, settlement=dt(2025, 2, 14), metric="duration")
   fxb.duration(ytm=4.0, settlement=dt(2025, 2, 14), metric="modified")
   fxb.duration(ytm=4.0, settlement=dt(2025, 2, 14), metric="risk")

.. image:: _static/ust_10y.gif
  :alt: US Treasury example using the FixedRateBond class
  :width: 611

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_instruments.rst

Calibrating curves with a solver
=================================

The guide for :ref:`Constructing Curves<curves-doc>` introduces the main
curve classes,
:class:`~rateslib.curves.Curve`, :class:`~rateslib.curves.LineCurve`, and
:class:`~rateslib.curves.IndexCurve`. It also touches on some of the more
advanced curves :class:`~rateslib.curves.CompositeCurve`,
and :class:`~rateslib.curves.ProxyCurve`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_curves.rst

Calibrating curves is a very natural thing to do in fixed income. We typically use
market prices of commonly traded instruments to set values.

Below we demonstrate how to calibrate the :class:`~rateslib.curves.Curve` that
we created above in the initial trivial example using SOFR swap market data. First, we
are reminded of the discount factors (DFs) currently set on that curve.

.. ipython:: python

   usd_curve.nodes

Now we will instruct a :class:`~rateslib.solver.Solver` to recalibrate those value to match
a set of instrument prices, ``s``.

.. ipython:: python

   from rateslib.solver import Solver
   usd_args = dict(
       effective=dt(2022, 1, 1),
       spec="usd_irs",
       curves="sofr"
   )
   solver = Solver(
       curves=[usd_curve],
       instruments=[
           IRS(**usd_args, termination="6M"),
           IRS(**usd_args, termination="1Y"),
       ],
       s=[4.35, 4.85],
       instrument_labels=["6M", "1Y"],
       id="us_rates"
   )

Solving was a success! Observe the DFs on the *Curve* have been updated:

.. ipython:: python

   usd_curve.nodes

We can plot the overnight rates for the calibrated curve. This curve uses *'log_linear'*
interpolation so the overnight forward rates are constant between node dates.

.. ipython:: python

   usd_curve.plot("1b", labels=["SOFR o/n"])

.. plot::

   from rateslib import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   usd_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 7, 1): 0.98,
           dt(2023, 1, 1): 0.95
       },
       calendar="nyc",
       id="sofr",
   )
   usd_args = dict(
       effective=dt(2022, 1, 1),
       spec="usd_irs",
       curves="sofr"
   )
   solver = Solver(
       curves=[usd_curve],
       instruments=[
           IRS(**usd_args, termination="6M"),
           IRS(**usd_args, termination="1Y"),
       ],
       s=[4.35, 4.85],
       instrument_labels=["6M", "1Y"],
       id="us_rates"
   )
   fig, ax, line = usd_curve.plot("1b", labels=["SOFR o/n"])
   plt.show()


Pricing Mechanisms
===================

Since *rateslib* is an object oriented library with object associations we give
detailed instructions of the way in which the associations can be constructed in
:ref:`mechanisms<mechanisms-doc>`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_mechanisms.rst


Risk Sensitivities
===================

Probably *rateslib's* main objective is to capture delta and gamma risks in a
generalised and holistic mathematical framework. See the
:ref:`risk framework<risk-toc-doc>` notes.


.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_risk.rst

Utilities
==========

*Rateslib* could not function without some utility libraries. These are often
referenced in other guides as they arise and can also be linked to from those
sections.

Specifically those utilities are:

- :ref:`Holiday calendars and day count conventions<cal-doc>`
- :ref:`Schedule building<schedule-doc>`
- :ref:`Piecewise polynomial splines for curve interpolation<splines-doc>`
- :ref:`Forward mode automatic differentiation (AD) and Dual numbers<dual-doc>`
- :ref:`Defaults used for instrument specification<defaults-doc>`

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_utilities.rst

Cookbook
=========

This is a collection of more detailed examples and explanations that don't necessarily fall
into any one category. Go to the :ref:`Cookbook index <cookbook-doc>`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_cookbook.rst


.. toctree::
    :hidden:

    g_coverage.rst