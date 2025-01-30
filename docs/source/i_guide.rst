.. RatesLib documentation master file,
.. _guide-doc:

==========
User Guide
==========

Where to start?
===============

*Rateslib* tends to follow the typical quant architecture:

.. image:: _static/instdatamodel.PNG
   :alt: Library pricing framework
   :align: center
   :width: 291

This means that **financial instrument specification**, **curve and/or surface construction**
from market data including **foreign exchange (FX)** will permit **pricing metrics** and **risk sensitivity**.
These functionalities are interlinked and potentially dependent upon each
other. This guide will introduce them in a structured way and give typical examples how they
are used in practice.

.. |ico3| image:: _static/rlxl32.png
   :height: 20px

.. note::

   If you see this icon |ico3| at any point after a section it will link to a section in the
   *rateslib-excel* documentation which may demonstrate the equivalent Python example in Excel.

Let's start with some fundamental *Curve* and *Instrument* constructors.

Trivial derivatives examples
----------------------------

*Rateslib* has two fundamental :ref:`Curve types<c-curves-doc>`. Both can be constructed
independently by providing basic inputs.

.. tabs::

   .. tab:: Curve

      A :class:`~rateslib.curves.Curve` is discount factor (DF) based and is constructed
      by providing DFs on specific node dates. Interpolation between ``nodes`` is
      configurable, but the below uses the *"log-linear"* default.

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

   .. tab:: LineCurve

      A :class:`~rateslib.curves.LineCurve` is value based and is constructed
      by providing curve values on specific node dates. Interpolation between ``nodes`` is
      configurable, with the default being *"linear"* interpolation (hence *LineCurve*).

      .. ipython:: python

         usd_legacy_3mIBOR = LineCurve(
             nodes={
                 dt(2022, 1, 1): 2.635,
                 dt(2022, 7, 1): 2.896,
                 dt(2023, 1, 1): 2.989,
             },
             calendar="nyc",
             id="us_ibor_3m",
         )

   .. tab:: *(IndexCurve)*

      An *Index Curve* is required by certain products, e.g.
      inflation linked bonds (:class:`~rateslib.instruments.IndexFixedRateBond`) or
      zero coupon inflation swaps (:class:`~rateslib.instruments.ZCIS`). Adding ``index_base``
      and ``index_lag`` argument inputs extends a DF based *Curve* to work in these cases.

      .. ipython:: python

         usd_cpi = Curve(
             nodes={
                 dt(2022, 1, 1): 1.00,
                 dt(2022, 7, 1): 0.97,
                 dt(2023, 1, 1): 0.955,
             },
             index_base=308.95,
             index_lag=3,
             interpolation="linear_index",
             id="us_cpi",
         )

   .. tab:: *(Hazard Curve)*

      A *Hazard Curve* is used by credit *Instruments* when a default is possible. *Hazard Curves*
      utilise the same :class:`~rateslib.curves.Curve` class and the rates reflect overnight
      hazard rates and the DFs reflect survival probabilities.

      .. ipython:: python

         pfizer_hazard = Curve(
             nodes={
                 dt(2022, 1, 1): 1.0,
                 dt(2022, 7, 1): 0.998,
                 dt(2023, 1, 1): 0.995
             },
             id="pfizer_hazard",
         )


Next, we will construct some basic derivative :ref:`Instruments<instruments-toc-doc>`.
These will use some market conventions defined by *rateslib* through its
:ref:`default argument specifications <defaults-arg-input>`, although all arguments can
be supplied manually if and when required.

.. tabs::

   .. group-tab:: IRS

      Here we create a short dated SOFR RFR interest rate swap (:class:`~rateslib.instruments.IRS`).

      .. ipython:: python

         irs = IRS(
             effective=dt(2022, 2, 15),
             termination="6m",
             notional=1000000000,
             fixed_rate=2.0,
             spec="usd_irs"
         )

   .. group-tab:: STIRFuture

      Here we create a SOFR STIR future (:class:`~rateslib.instruments.STIRFuture`).

      .. ipython:: python

         stir = STIRFuture(
             effective=get_imm(code="H22"),
             termination=get_imm(code="M22"),
             contracts=100,
             price=97.495,
             spec="usd_stir",
         )

   .. group-tab:: FRA

      A US LIBOR :class:`~rateslib.instruments.FRA` is an obsolete *Instrument*, but we can still
      create one and these still trade in other currencies, e.g. EUR.

      .. ipython:: python

         fra = FRA(
             effective=dt(2022, 2, 16),
             termination="3m",
             frequency="Q",
             calendar="nyc",
             convention="act360",
             method_param=2,
             fixed_rate=2.5
         )

   .. group-tab:: CDS

      Here we construct a generic US investment grade credit default
      swap (:class:`~rateslib.instruments.CDS`)

      .. ipython:: python

         cds = CDS(
             effective=dt(2021, 12, 20),
             termination=dt(2022, 9, 20),
             notional=15e6,
             spec="us_ig_cds",
         )

   .. group-tab:: ZCIS

      This constructs a zero-coupon inflation swap
      (:class:`~rateslib.instruments.ZCIS`) with usual
      daily index interpolation and 3-month index lag.

      .. ipython:: python

         zcis = ZCIS(
             effective=dt(2022, 2, 2),
             termination="9m",
             notional=-25e6,
             fixed_rate=3.25,
             spec="usd_zcis",
         )

We can combine the *Curves* and the *Instruments* to give pricing metrics such as
:meth:`~rateslib.instruments.BaseDerivative.npv`,
:meth:`~rateslib.instruments.BaseDerivative.cashflows`, and the mid-market
:meth:`~rateslib.instruments.BaseDerivative.rate`, as well as others. Without further specification
these values are all expressed in the *Instrument's* local USD currency.

.. tabs::

   .. group-tab:: IRS

      .. ipython:: python

         irs.npv(usd_curve)

      .. ipython:: python

         irs.cashflows(usd_curve)

   .. group-tab:: STIRFuture

      .. ipython:: python

         stir.npv(usd_curve)

      .. ipython:: python

         stir.rate(usd_curve, metric="price")

   .. group-tab:: FRA

      .. ipython:: python

         fra.npv(curves=[usd_legacy_3mIBOR, usd_curve])

      .. ipython:: python

         fra.rate([usd_legacy_3mIBOR, usd_curve])

   .. group-tab:: CDS

      .. ipython:: python

         cds.npv(curves=[pfizer_hazard, usd_curve])

      .. ipython:: python

         cds.rate([pfizer_hazard, usd_curve])

      .. ipython:: python

         cds.cashflows([pfizer_hazard, usd_curve])

   .. group-tab:: ZCIS

      .. ipython:: python

         zcis.npv([usd_cpi, usd_curve])

      .. ipython:: python

         zcis.rate([usd_cpi, usd_curve])

      .. ipython:: python

         zcis.cashflows([usd_cpi, usd_curve])

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_introduction.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel introductory example" width="20">
     </a>
   </div>

If instead of this trivial, minimalist example you would like to see a real world
example :ref:`replicating a Bloomberg SWPM function SOFR curve<cook-swpm-doc>` please
click the link.


Quick look at FX
==================

Spot rates and conversion
-------------------------

The above values were all calculated and displayed in USD. That is the default
currency in *rateslib* and the local currency of those *Instruments*. We can convert these values
into another currency using the :class:`~rateslib.fx.FXRates` class. This is a basic class which is
parametrised by some exchange rates.

.. tabs::

   .. tab:: FXRates

      .. ipython:: python

         fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.25})
         fxr.rates_table()

We now have a mechanism by which to specify values in other currencies.

.. tabs::

   .. group-tab:: IRS

      .. ipython:: python

         irs.npv(usd_curve, fx=fxr, base="usd")
         irs.npv(usd_curve, fx=fxr, base="eur")

   .. group-tab:: STIRFuture

      .. ipython:: python

         stir.npv(usd_curve, fx=fxr, base="usd")
         stir.npv(usd_curve, fx=fxr, base="eur")

   .. group-tab:: FRA

      .. ipython:: python

         fra.npv([usd_legacy_3mIBOR, usd_curve], fx=fxr, base="usd")
         fra.npv([usd_legacy_3mIBOR, usd_curve], fx=fxr, base="eur")

   .. group-tab:: CDS

      .. ipython:: python

         cds.npv([pfizer_hazard, usd_curve], fx=fxr, base="usd")
         cds.npv([pfizer_hazard, usd_curve], fx=fxr, base="eur")

   .. group-tab:: ZCIS

      .. ipython:: python

         zcis.npv([usd_cpi, usd_curve], fx=fxr, base="usd")
         zcis.npv([usd_cpi, usd_curve], fx=fxr, base="eur")

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_introduction_fx.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel introductory example" width="20">
     </a>
   </div>

One observes that the value returned here is not a float but a :class:`~rateslib.dual.Dual`
which is part of *rateslib's* AD framework. This is the first example of capturing a
sensitivity, which here denotes the sensitivity of the EUR NPV relative to the EURUSD FX rate.
One can read more about this particular treatment of FX
:ref:`here<fx-dual-doc>` and more generally about the dual AD framework :ref:`here<dual-doc>`.

FX forwards
------------

For multi-currency derivatives we need more than basic, spot exchange rates.
We need an :class:`~rateslib.fx.FXForwards` market.
This stores the FX rates and the interest
rates curves that are used for all the FX-interest rate parity derivations. With these
we can calculate forward FX rates and also ad-hoc FX swap rates.

When defining the ``fx_curves`` dict mapping, the key *"eurusd"* should be interpreted as; **the
Curve for EUR cashflows, collateralised in USD**, and similarly for other entries.

.. tabs::

   .. tab:: FXForwards

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
         fxf.swap("eurusd", settlements=[dt(2022, 2, 1), dt(2022, 5, 2)])

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_introduction_fx_forwards.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel FXForwards introduction" width="20">
     </a>
   </div>

*FXForwards* objects are comprehensive and more information regarding all of the
:ref:`FX features<fx-doc>` is available in this link.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_fx.rst

More about instruments
======================

We've seen some *single currency derivatives* above.
A complete guide for all of the :ref:`Instruments<instruments-toc-doc>` is available in
this link. That will also introduce the building blocks; :ref:`Legs<legs-doc>` and :ref:`Periods<periods-doc>`.

Multi-currency instruments
--------------------------

Let's take a look at the multi-currency instruments. Notice how these *Instruments*
maintain consistent method naming conventions with those above. This makes it possible to plug
any *Instruments* into a :class:`~rateslib.solver.Solver` to calibrate *Curves*
around target mid-market rates, and generate market risks.

.. tabs::

   .. tab:: FXSwap

      This is an :class:`~rateslib.instruments.FXSwap`.

      .. ipython:: python

         fxs = FXSwap(
             effective=dt(2022, 2, 1),
             termination="3m",  # May-1 is a holiday, May-2 is business end date.
             pair="eurusd",
             notional=20e6,
             calendar="tgt|fed",
         )
         fxs.rate(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)
         fxs.cashflows_table(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)

   .. tab:: FXExchange

      An :class:`~rateslib.instruments.FXExchange` is a forward FX transaction.

      .. ipython:: python

         fxe = FXExchange(
             settlement=dt(2022, 4, 1),
             pair="eurusd",
             notional=10e6,
             fx_rate=1.035,
         )
         fxe.rate(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)
         fxe.npv(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)
         fxe.cashflows_table(curves=[None, eurusd_curve, None, usd_curve], fx=fxf)

   .. tab:: XCS

      Cross-currency swaps (:class:`~rateslib.instruments.XCS`) are easily configured and
      analysed in *rateslib*.

      .. ipython:: python

         xcs = XCS(
             effective=dt(2022, 4, 1),
             termination="6m",
             spec="eurusd_xcs",
             float_spread=-3.0,
             notional=25e6,
         )
         xcs.rate(curves=[eur_curve, eurusd_curve, usd_curve, usd_curve], fx=fxf)
         xcs.cashflows(curves=[eur_curve, eurusd_curve, usd_curve, usd_curve], fx=fxf)
         xcs.cashflows_table(curves=[eur_curve, eurusd_curve, usd_curve, usd_curve], fx=fxf)

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_fxswap_intro.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel FXSwap introduction" width="20">
     </a>
   </div>

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

.. raw:: html

   <div class="clear" style="padding-bottom: 1em;"></div>

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_bond_intro.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel FixedRateBond introduction" width="20">
     </a>
   </div>

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
curve classes, :class:`~rateslib.curves.Curve` and :class:`~rateslib.curves.LineCurve`.
It also touches on some of the more
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

   solver = Solver(
       curves=[usd_curve],
       instruments=[
           IRS(dt(2022, 1, 1), "6M", spec="usd_irs", curves="sofr"),
           IRS(dt(2022, 1, 1), "1Y", spec="usd_irs", curves="sofr"),
       ],
       s=[4.35, 4.85],
       instrument_labels=["6M", "1Y"],
       id="us_rates"
   )

.. raw:: html

   <div style="width: 100%; padding: 0em 0em 1em; text-align: center;">
     <a href="https://rateslib.com/excel/latest/z_introduction_solver.html" target="_blank">
       <img src="_static/rlxl32.png" alt="Rateslib-excel Solver introduction" width="20">
     </a>
   </div>

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

Risk Sensitivities
===================

*Rateslib's* can calculate **delta** and **cross-gamma** risks relative to the calibrating
*Instruments* of a *Solver*. Rateslib also unifies these risks against the **FX rates**
used to create an *FXForwards* market, to provide a fully consistent risk framework
expressed in arbitrary currencies. See the
:ref:`risk framework<risk-toc-doc>` notes.

Performance wise, because *rateslib* uses dual number AD upto 2nd order, combined with the
appropriate analysis, it is shown to calculate a 150x150 *Instrument* cross-gamma grid
(22,500 elements) from a calculated portfolio NPV in approximately 250ms.

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

Advanced Concepts
==================

These sections describe and exemplify some of the architectural choices in *rateslib*.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_advancedconcepts.rst

Pricing Mechanisms
------------------

Since *rateslib* is an object oriented library with object associations we give
detailed instructions of the way in which the associations can be constructed in
:ref:`mechanisms<mechanisms-doc>`.

The **key takeway** is that when you initialise and create an *Instrument* you can do one
of three things:

1) Not provide any *Curves* (or *Vol* surface) for pricing upfront (``curves=NoInput(0)``).
2) Create an explicit association to pre-existing Python objects, e.g. ``curves=my_curve``.
3) Define some reference to a *Curves* mapping with strings using ``curves="my_curve_id"``.

If you do *1)* then you must provide *Curves* at price
time: ``instrument.npv(curves=my_curve)``.

If you do *2)* then you do not need to provide anything further at price time:
``instrument.npv()``. But you still can provide *Curves* directly, like for *1)*, as an override.

If you do *3)* then you can provide a :class:`~rateslib.solver.Solver` which contains the *Curves* and will
resolve the string mapping: ``instrument.npv(solver=my_solver)``. But you can also provide *Curves*
directly, like for *1)*, as an override.

**Best practice** in *rateslib* is to use *3)*. This is the safest and most flexible approach and
designed to work best with risk sensitivity calculations also.

Mutability
----------

A proper outline of the mutability of objects is given in :ref:`mutability<mutability-doc>`.

In summary, **best practice** is to create new instances and avoid directly overwriting or adding
to class attributes. *Don't mutate a created object unless using an official method to do so*,
e.g. :meth:`FXRates.update<rateslib.fx.FXRates.update>` or
:meth:`Curve.update<rateslib.curves.Curve.update>`


.. toctree::
    :hidden:

    g_coverage.rst
