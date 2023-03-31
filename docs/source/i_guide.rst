.. RatesLib documentation master file,

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

For example, we can construct a :class:`~rateslib.curves.Curve` in a number of ways:
here by direct specification of discount factors (DFs).

.. ipython:: python

   from rateslib.curves import Curve
   curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 7, 1): 0.98,
           dt(2023, 1, 1): 0.95
       },
       calendar="nyc",
   )

We can construct an ``Instrument``: here a short dated RFR interest rate swap
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

We can value the IRS in its local currency by default (USD) or in another
currency by creating an
:class:`~rateslib.fx.FXRates` object and putting everything together.

.. ipython:: python

   irs.npv(curve)

We can obtain basic IR delta risk sensitivity, expressed in local currency,
for this IRS with the basic :meth:`~rateslib.instruments.IRS.analytic_delta`
method.

.. ipython:: python

   irs.analytic_delta(curve)

We cannot, in this example, obtain more precise (bucket) risk sensitivities of this
IRS, since, in order to do so,
we need to configure a risk framework and construct curves from instruments
using a :class:`~rateslib.solver.Solver`.

Reading the Guide
==================

It is suggested that the guide is read in the following order.

FX
--

First, start with :ref:`FX<fx-doc>`, since this gives a nice introduction
of a class that can be used independently.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    g_fx.rst

FX objects allow the expression of values converted from one currency to another.

.. ipython:: python

   from rateslib.fx import FXRates
   fxr = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 1))
   irs.npv(curve, None, fxr, "eur")

Instruments
-----------

Next move on to reviewing the :ref:`instruments<instruments-toc-doc>`. This gives
an overview of the financial products that ``rateslib`` can currently price. It is
recommended to review :ref:`periods<periods-doc>` and then :ref:`legs<legs-doc>`, since
these building blocks and their documentation provides technical descriptions of the
parameters that can be used.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    g_instruments.rst

Curves and Solver
------------------

The guide for :ref:`curves<curves-doc>` introduces the two main curve classes,
:class:`~rateslib.curves.Curve` and :class:`~rateslib.curves.LineCurve`. It gives
examples of their parametrisation and different interpolation structures. A
simple curve was created above, and often in many ``rateslib`` examples the curves
creates are simple to exemplify other features. The documentation here explains the
nuances of curves.

.. toctree::
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

Since ``rateslib`` is an object oriented library with object associations we give
detailed instructions of the way in which the associations can be constructed in
:ref:`mechanisms<mechanisms-doc>`.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    g_mechanisms.rst


Risk Sensitivities
-------------------

TBD

Utilities
----------

``rateslib`` could not function without some utility libraries. These are often
referenced in other guides as they come up and can also be linked to from those
sections.

.. toctree::
    :maxdepth: 0
    :titlesonly:

    g_utilities.rst

Coverage
--------

The current test coverage status of ``rateslib`` is shown at XXXXXX %.

.. toctree::
    :hidden:

    g_coverage.rst