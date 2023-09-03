.. RatesLib documentation master file,
.. _guide-doc:

==========
User Guide
==========

Where to Start?
===============

It is important to understand that the key elements of this library are
**curve construction**, **financial instrument specification**,
**foreign exchange (FX)** and **risk sensitivity**.
All of these functionalities are interlinked and potentially dependent upon each
other. This guide's intention is to introduce them in a structured way.

Let's start with a basic *Curve* and *Instrument*.

A Trivial Minimalist Example
----------------------------

For example, we can construct :ref:`Curves<c-curves-doc>` in many different ways:
here we create one by directly specifying discount factors (DFs) on certain node dates.

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
   )

We can then construct an :ref:`Instrument<instruments-toc-doc>`. Here we create a short dated
RFR interest rate swap (:class:`~rateslib.instruments.IRS`) using market specification pre-defined
by *rateslib*. You can read more about arguments for instruments :ref:`here<defaults-doc>`.

.. ipython:: python

   from rateslib.instruments import IRS
   irs = IRS(
       effective=dt(2022, 2, 15),
       termination="6m",
       notional=1000000000,
       fixed_rate=2.0,
       spec="usd_irs"
   )

We can value the IRS in its local currency (USD) by default, and see the generated
cashflows.

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

FX Forwards
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

More about Instruments
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
       curves=[None, eurusd_curve, None, usd_curve]
   )
   fxs.rate(fx=fxf)

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

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_instruments.rst

Combining Curves and Solver
============================

.. The *Curves* and *Solver* contained within *rateslib* are completely
   generalised. Behind the scenes the same basic optimisation routine is utilised for any
   complex structure one can craft that is mathematically valid

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

We can see an example of a *ProxyCurve* if we observe that the *FXForwards* class
we created was derived by EUR cashflows collateralised in USD. That object can
auto-generate the discount factors for USD cashflows collateralised in EUR:

.. ipython:: python

   fxf.curve(cashflow="usd", collateral="eur")

For derivatives collateralised according to a multi-currency CSA this
curve can also be auto-generated and it results in an intrinsic composition
of the *"usdeur* and *usdusd* curves.

.. ipython:: python

   fxf.curve(cashflow="usd", collateral=["eur", "usd"])

The :class:`~rateslib.solver.Solver` is an advanced global optimiser which can
solve one or any number of curves simultaneously. *Solver* can even be combined in a
dependency chain so that ceratin curves might be solved before others, which is common
in a fixed income trading team with segregated responsibilities.

We have so far defined 3 base curves which each have 2 degrees of freedom. In order to
solver this system we need 6 market instruments. These will be 2 *IRS* for controlling
the EUR curve, 2 *IRS* for the USD curve, and 2 *FXSwaps* for controlling the basis
curve.

.. ipython:: python

   solver = Solver(
       curves=[eur_curve, usd_curve, eurusd_curve],
       instruments=[
           IRS(dt(2022, 1, 1), "6M", "A", calendar="tgt", currency="eur", curves=eur_curve),
           IRS(dt(2022, 1, 1), "1Y", "A", calendar="tgt", currency="eur", curves=eur_curve),
           IRS(dt(2022, 1, 1), "6M", "A", calendar="nyc", currency="usd", curves=usd_curve),
           IRS(dt(2022, 1, 1), "1Y", "A", calendar="nyc", currency="usd", curves=usd_curve),
           FXSwap(dt(2022, 1, 1), "6M", currency="eur", leg2_currency="usd", curves=[None, eurusd_curve, None, usd_curve]),
           FXSwap(dt(2022, 1, 1), "1Y", currency="eur", leg2_currency="usd", curves=[None, eurusd_curve, None, usd_curve]),
       ],
       s=[2.25, 2.5, 4.5, 5.0, 30, 75],
       instrument_labels=["Eur 6M", "Eur 1Y", "Usd 6M", "Usd 1Y", "EurUsd 6M", "EurUsd 1Y"],
       fx=fxf,
   )

The *Curves* have all been dynamically updated according to the *Solver's*
optimisation routines. We can plots these resultant curves.

.. ipython:: python

   usd_curve.plot(
       "1b",
       comparators=[fxf.curve("usd", "eur"), eur_curve, eurusd_curve],
       labels=["usd:usd", "usd:eur", "eur:eur", "eur:usd"]
   )

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
   )
   eur_curve = Curve({
       dt(2022, 1, 1): 1.0,
       dt(2022, 7, 1): 0.972,
       dt(2023, 1, 1): 0.98}
   )
   eurusd_curve = Curve({
       dt(2022, 1, 1): 1.0,
       dt(2022, 7, 1): 0.973,
       dt(2023, 1, 1): 0.981}
   )
   fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 1))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "usdusd": usd_curve,
           "eureur": eur_curve,
           "eurusd": eurusd_curve,
       }
   )
   fxf.rate("eurusd", settlement=dt(2023, 1, 1))
   solver = Solver(
       curves=[eur_curve, usd_curve, eurusd_curve],
       instruments=[
           IRS(dt(2022, 1, 1), "6M", "A", calendar="tgt", currency="eur", curves=eur_curve),
           IRS(dt(2022, 1, 1), "1Y", "A", calendar="tgt", currency="eur", curves=eur_curve),
           IRS(dt(2022, 1, 1), "6M", "A", calendar="nyc", currency="usd", curves=usd_curve),
           IRS(dt(2022, 1, 1), "1Y", "A", calendar="nyc", currency="usd", curves=usd_curve),
           FXSwap(dt(2022, 1, 1), "6M", currency="eur", leg2_currency="usd", curves=[None, eurusd_curve, None, usd_curve]),
           FXSwap(dt(2022, 1, 1), "1Y", currency="eur", leg2_currency="usd", curves=[None, eurusd_curve, None, usd_curve]),
       ],
       s=[2.25, 2.5, 4.5, 5.0, 30, 75],
       instrument_labels=["Eur 6M", "Eur 1Y", "Usd 6M", "Usd 1Y", "EurUsd 6M", "EurUsd 1Y"],
       fx=fxf,
   )
   fig, ax, line = usd_curve.plot("1b", comparators=[fxf.curve("usd", "eur"), eur_curve, eurusd_curve], labels=["usd:usd", "usd:eur", "eur:eur", "eur:usd"])
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
referenced in other guides as they come up and can also be linked to from those
sections.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_utilities.rst

Coverage
==========

The current test coverage status of *rateslib* is shown at around 97%.

.. toctree::
    :hidden:

    g_coverage.rst