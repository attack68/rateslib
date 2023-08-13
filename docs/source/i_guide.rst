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

For example, we can construct a :ref:`Curve<c-curves-doc>` in a number of ways:
here by direct specification of discount factors (DFs).

.. ipython:: python

   from rateslib.curves import Curve
   usd_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 7, 1): 0.98,
           dt(2023, 1, 1): 0.95
       },
       calendar="nyc",
   )

We can construct an :ref:`Instrument<instruments-toc-doc>`: here a short dated RFR interest rate swap
(:class:`~rateslib.instruments.IRS`).

.. ipython:: python

   from rateslib.instruments import IRS
   irs = IRS(
       effective=dt(2022, 2, 15),
       termination="6m",
       frequency="M",
       currency="usd",
       calendar="nyc",
       fixed_rate=2.0,
       notional=1000000000,
   )

We can value the IRS in its local currency (USD) by default, and see the generated
cashflows.

.. ipython:: python

   irs.npv(usd_curve)

.. ipython:: python

   irs.cashflows(usd_curve)

But this is only the the most basic functionality *rateslib* offers. Other features
that will be explained below are:

- how to parametrize ``Curves`` and how to calibrate them to market rates
  using a :class:`~rateslib.solver.Solver`,
- the various different ``Instruments`` that are offered and the various methods
  used to explore their construction, e.g. ``cashflows()``.
- how ``FX`` is handled and how any value can be converted into another currency
  **preserving fx rate sensitivity**.
- how **risk sensitivity** is addressed.

Reading the Guide
==================

It is suggested that the guide is read in the following order.

FX
--

First, start with :ref:`FX<fx-doc>`, since this gives a nice introduction
of classes that can be used independently.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_fx.rst

:class:`~rateslib.fx.FXRates` objects allow the expression of values
converted from one currency to another.

.. ipython:: python

   from rateslib.fx import FXRates, FXForwards
   fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 1))
   irs.npv(usd_curve, None, fxr, "eur")

With a little more construction, :class:`~rateslib.fx.FXForwards` objects allow the
calculation of FX forward rates under a no arbitrage, cash-collateral consistent
framework. These objects are required to price mark-to-market (MTM) multi-currency
derivatives.

.. ipython:: python

   eur_curve = Curve({dt(2022, 1, 1): 1.0, dt(2024, 1, 1): 0.98})
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "usdusd": usd_curve,
           "eureur": eur_curve,
           "eurusd": eur_curve,
       }
   )
   fxf.rate("eurusd", settlement=dt(2023, 1, 1))

Instruments
-----------

Next move on to reviewing the :ref:`Instruments<instruments-toc-doc>`. This gives
an overview of the financial products that *rateslib* can currently price. It is
recommended to review :ref:`Periods<periods-doc>` and then :ref:`Legs<legs-doc>`, since
the documentation for these building blocks provides technical descriptions of the
parameters that can be used.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_instruments.rst

Curves and Solver
------------------

The guide for :ref:`Constructing Curves<curves-doc>` introduces the two main
curve classes,
:class:`~rateslib.curves.Curve` and :class:`~rateslib.curves.LineCurve`. It gives
examples of their parametrization and different interpolation structures. A
simple curve was created above, and often in many *rateslib* examples the curves
created are simple to exemplify other features. The documentation here explains the
nuances of curves.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_curves.rst

Once all of the information about :ref:`FX<fx-doc>`,
:ref:`instruments<instruments-toc-doc>`, and :ref:`curves<curves-doc>` is
understood you will be in a good position to build :class:`~rateslib.solver.Solver`
frameworks, which are a completely general solution for calibrating curves
given a set of mid-market prices.

Pricing Mechanisms
-------------------

Since *rateslib* is an object oriented library with object associations we give
detailed instructions of the way in which the associations can be constructed in
:ref:`mechanisms<mechanisms-doc>`.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_mechanisms.rst


Risk Sensitivities
-------------------

Probably *rateslib's* main objective is to capture delta and gamma risks in a
generalised and holistic mathematical framework. See the
:ref:`risk framework<risk-toc-doc>` notes.


.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_risk.rst

Utilities
----------

*Rateslib* could not function without some utility libraries. These are often
referenced in other guides as they come up and can also be linked to from those
sections.

.. toctree::
    :hidden:
    :maxdepth: 0
    :titlesonly:

    g_utilities.rst

Coverage
--------

The current test coverage status of *rateslib* is shown at around 97%.

.. toctree::
    :hidden:

    g_coverage.rst