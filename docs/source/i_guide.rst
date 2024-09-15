.. RatesLib documentation master file,
.. _guide-doc:

==========
User Guide
==========

Where to start?
===============

It is important to understand that this library tends to follow the typical framework:

.. image:: _static/instdatamodel.PNG
   :alt: Library pricing framework
   :align: center
   :width: 291

This means that **financial instrument specification**, **curve and/or surface construction**
from market data including **foreign exchange (FX)** will permit **pricing metrics** and **risk sensitivity**.
These functionalities are interlinked and potentially dependent upon each
other. This guide's intention is to introduce them in a structured way and give typical examples how they
are used in practice.

.. |ico3| image:: _static/rlxl32.png
   :height: 20px

.. note::

   If you see this icon |ico3| at any point after a section it will link to a section in the
   *rateslib-excel* documentation which may demonstrate the equivalent Python example in Excel.

Let's start with the fundamental constructors *Curve* and *Instrument*.

A trivial example
----------------------------

For example, we can construct :ref:`Curves<c-curves-doc>` in many different ways:
here we create one by directly specifying discount factors (DFs) on certain node dates (sometimes
called pillar dates in other publications).

.. ipython:: python

   from rateslib import *

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
RFR interest rate swap (:class:`~rateslib.instruments.IRS`) using the conventional market specification
*rateslib* offers many examples of *Instrument* specifications as all seen
:ref:`here in defaults <defaults-arg-input>`.

.. ipython:: python

   irs = IRS(
       effective=dt(2022, 2, 15),
       termination="6m",
       notional=1000000000,
       fixed_rate=2.0,
       spec="usd_irs"
   )

We can value the *IRS* with the *Curve* in its local currency (USD) by default, and see
the generated cashflows.

.. ipython:: python

   irs.npv(usd_curve)

.. ipython:: python

   irs.cashflows(usd_curve)

.. image:: _static/rlxl32.png
  :align: center
  :alt: Rateslib-excel introductory example
  :width: 20
  :target: https://rateslib.com/excel/latest/z_introduction.html

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

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

   fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.25})
   fxr.rates_table()

We now have a mechanism by which to specify values in other currencies.

.. ipython:: python

   irs.npv(usd_curve, fx=fxr, base="usd")
   irs.npv(usd_curve, fx=fxr, base="eur")

.. image:: _static/rlxl32.png
  :align: center
  :alt: Rateslib-excel introductory example
  :width: 20
  :target: https://rateslib.com/excel/latest/z_introduction_fx.html

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

One observes that the value returned here is not a float but a :class:`~rateslib.dual.Dual`
which is part of *rateslib's* AD framework. This is the first example of capturing a
sensitivity, which here denotes the sensitivity of the EUR NPV relative to the EURUSD FX rate.
One can read more about this particular treatment of FX
:ref:`here<fx-dual-doc>` and more generally about the dual AD framework :ref:`here<dual-doc>`.

FX forwards
------------

For multi-currency derivatives we need more than basic, spot exchange rates.
We can also create an
:class:`~rateslib.fx.FXForwards` class. This stores the FX rates and the interest
rates curves that are used for all the FX-interest rate parity derivations. With these
we can calculate forward FX rates and also ad-hoc FX swap rates.

When defining the ``fx_curves`` dict mapping, the key *"eurusd"* should be interpreted as; **the
Curve for EUR cashflows, collateralised in USD**, and similarly for other entries.

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

.. image:: _static/rlxl32.png
  :align: center
  :alt: Rateslib-excel FXForwards introduction
  :width: 20
  :target: https://rateslib.com/excel/latest/z_introduction_fx_forwards.html

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

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

Let's take a look at an example of a multi-currency instrument: the
:class:`~rateslib.instruments.FXSwap`. All instruments have a mid-market pricing
function :meth:`rate()<rateslib.instruments.BaseDerivative.rate>`. Keeping a
consistent function name across all *Instruments* allows any of them to be used within a
:class:`~rateslib.solver.Solver` to calibrate *Curves* around target mid-market rates.

This *FXSwap* *Instrument* construction prices to the same mid-market rate as the ad-hox *swap* rate
used in the example above (as expected).

.. ipython:: python

   fxs = FXSwap(
       effective=dt(2022, 2, 1),
       termination="3m",  # May-1 is a holiday, May-2 is business end date.
       pair="eurusd",
       notional=20e6,
       calendar="tgt|fed",
   )
   fxs.rate(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)

.. image:: _static/rlxl32.png
  :align: center
  :alt: Rateslib-excel FXSwap introduction
  :width: 20
  :target: https://rateslib.com/excel/latest/z_fxswap_intro.html

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

Securities and bonds
--------------------

A very common instrument in financial investing is a :class:`~rateslib.instruments.FixedRateBond`.
At time of writing the on-the-run 10Y US treasury was the 3.875% Aug 2033 bond. Here we can
construct this using the street convention and derive the price from yield-to-maturity and
risk calculations.

.. ipython:: python

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

.. image:: _static/rlxl32.png
  :align: center
  :alt: Rateslib-excel FixedRateBond introduction
  :width: 20
  :target: https://rateslib.com/excel/latest/z_bond_intro.html

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_instruments.rst

There are some interesting :ref:`Cookbook <cookbook-doc>` articles
on :class:`~rateslib.instruments.BondFuture` and cheapest-to-deliver (CTD) analysis.

Calibrating curves with a solver
=================================

The guide for :ref:`Constructing Curves<curves-doc>` introduces the main
curve classes,
:class:`~rateslib.curves.Curve`, :class:`~rateslib.curves.LineCurve`, and
:class:`~rateslib.curves.IndexCurve`. It also touches on some of the more
advanced curves :class:`~rateslib.curves.CompositeCurve`,
:class:`~rateslib.curves.ProxyCurve`, and :class:`~rateslib.curves.MultiCsaCurve`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_curves.rst

Calibrating curves is a very natural thing to do in fixed income. We typically use
market prices of commonly traded instruments to set values. *FX Volatility Smiles* and
*FX Volatility Surfaces* are also calibrated using the exact same optimising algorithms.

Below we demonstrate how to calibrate the :class:`~rateslib.curves.Curve` that
we created above in the initial trivial example using SOFR swap market data. First, we
are reminded of the discount factors (DFs) which were manually set on that curve.

.. ipython:: python

   usd_curve.nodes

Now we will instruct a :class:`~rateslib.solver.Solver` to recalibrate those value to match
a set of prices, ``s``. The calibrating *Instruments* associated with those prices are 6M and 1Y *IRSs*.

.. ipython:: python

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

Solving was a success! Observe that the DFs on the *Curve* have been updated:

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

The **key takeway** is that when you initialise and create an *Instrument* you can do one
of three things:

1) Not provide any *Curves* (or *Vol* surface) for pricing upfront (``curves=NoInput(0)``).
2) Create an explicit association to pre-existing Python objects, e.g. ``curves=my_curve``.
3) Define some reference to a *Curves* mapping with strings using ``curves="my_curve_id"``.


If you do *1)* then you have to provide *Curves* at price
time: ``instrument.npv(curves=my_curve)``.

If you do *2)* then you do not need to provide anything further at price time:
``instrument.npv()``, or can provide new *Curves* directly, like for *1)*, as an override.

If you do *3)* then you can provide a :class:`~rateslib.solver.Solver` which contains the *Curves* and will
resolve the string mapping: ``instrument.npv(solver=my_solver)``. Or you can also provide *Curves*
directly, like for *1)*.

**Best practice** in *rateslib* is to use *3)*. This is the safest and most flexible approach and
designed to work best with risk sensitivity calculations also.


Risk Sensitivities
===================

*Rateslib's* can calculate **delta** and **cross-gamma** risks relative to the calibrating
*Instruments* of a *Solver*. Rateslib also unifies these risks against the **FX rates**
used to create an *FXForwards* market, to provide a fully consistent risk framework
expressed in arbitrary currencies. See the
:ref:`risk framework<risk-toc-doc>` notes.

Performance wise, because *rateslib* uses dual number AD upto 2nd order, combined with the
appropriate analysis, it is shown to calculate a 150x150 *Instrument* cross-gamma grid
(22,500 elements) from a calculated portfolio NPV in approximately 1 second.

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
into any one category. See the :ref:`Cookbook index <cookbook-doc>`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_cookbook.rst

.. toctree::
    :hidden:

    g_coverage.rst
