.. _pricing-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

***********
Get Started
***********

Installation
------------

*Rateslib* can **only** be installed on a machine under the terms of its :ref:`licence <licence-doc>`

.. raw:: html

   <p>If you would like to install rateslib on a corporate machine without a licence extension,
      for a trial period of time, please register your interest by sending an email
      to
      <span style="color:black; font-style: italic;font-weight: bold;">info@r<span class="spamoff">Stockholm Kungsgatan</span>ateslib.com</span>
      , who will grant such a request.
   </p>

*Rateslib* can be installed directly from
`PyPI <https://pypi.org/project/rateslib/#description>`_ using ``pip`` into your Python
environment.

.. code-block::

   pip install rateslib

Versions of *rateslib* greater than and starting at 1.2.0 use `Rust <https://www.rust-lang.org/>`_ extensions
for performance. For most users this will not affect the installation of *rateslib*, however for some
computer architectures (e.g. Linux) Python wheels are not pre-built, and this means ``pip install rateslib`` will
use the source distribution directly. In this case you must first
`install Rust <https://www.rust-lang.org/tools/install>`_ so that the rust extensions
can be compiled locally by the *pip* installer.

*Rateslib* is also available via the community ``conda-forge`` channel for
`Anaconda <https://anaconda.org/conda-forge/rateslib>`_, but only
for platforms Osx-64, Windows-64 and Linux-64. Other platform installs will default to the much
earlier pre-rust 1.1.0 version.

.. code-block::

   conda install --channel=conda-forge rateslib

**Minimum Dependencies**

.. list-table::
   :widths: 25 25 25 25
   :header-rows: 1


   * - Package Name
     - Latest Tested
     - Recommended Version
     - Earliest Tested
   * - Python
     - 3.14
     - 3.13
     - 3.10 (Oct '21)
   * - NumPy
     - 2.3.5
     - 2.2.6
     - 1.21.5 (Dec '21)
   * - Pandas
     - 2.3.3
     - 2.2.3
     - 1.4.1 (Feb '22)
   * - Matplotlib
     - 3.10.8
     - 3.9.4
     - 3.5.1 (Dec '21)


Introduction to Rateslib
-------------------------

For what purpose would I use *rateslib*?
=============================================

- If you want to integrate linear fixed income, FX and FX volatility analysis into your workflow with Python.
- If you desire a pain free setup process, a user-oriented API, and extensive documentation.
- If you are new to fixed income and currencies and interested to learn about basic and advanced concepts with
  tools to explore the nuances of these markets, as a companion to various authored books.

Which ``instruments`` does *rateslib* include?
===========================================================================

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1


   * - Single Ccy Derivatives
     - Multi-Ccy Derivatives
     - Securities
     - FX Volatility
     - Combinations
   * - :class:`~rateslib.instruments.IRS`
     - :class:`~rateslib.instruments.FXForward`
     - :class:`~rateslib.instruments.FixedRateBond`
     - :class:`~rateslib.instruments.FXCall`
     - :class:`~rateslib.instruments.Spread`
   * - :class:`~rateslib.instruments.SBS`
     - :class:`~rateslib.instruments.FXSwap`
     - :class:`~rateslib.instruments.FloatRateNote`
     - :class:`~rateslib.instruments.FXPut`
     - :class:`~rateslib.instruments.Fly`
   * - :class:`~rateslib.instruments.FRA`
     - :class:`~rateslib.instruments.XCS`
     - :class:`~rateslib.instruments.Bill`
     - :class:`~rateslib.instruments.FXRiskReversal`
     - :class:`~rateslib.instruments.Portfolio`
   * - :class:`~rateslib.instruments.STIRFuture`
     - :class:`~rateslib.instruments.NDF`
     - :class:`~rateslib.instruments.BondFuture`
     - :class:`~rateslib.instruments.FXStraddle`
     -
   * - :class:`~rateslib.instruments.ZCS`
     - ND- :class:`~rateslib.instruments.IRS`
     - :class:`~rateslib.instruments.IndexFixedRateBond`
     - :class:`~rateslib.instruments.FXStrangle`
     -
   * - :class:`~rateslib.instruments.ZCIS`
     - :class:`~rateslib.instruments.NDXCS`
     -
     - :class:`~rateslib.instruments.FXBrokerFly`
     -
   * - :class:`~rateslib.instruments.IIRS`
     -
     -
     -
     -
   * - :class:`~rateslib.instruments.CDS`
     -
     -
     -
     -

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<instruments-toc-doc>`

.. raw:: html

    </div>

Does *rateslib* handle ``foreign exchange (FX)``?
===========================================================

**Yes**. Foreign exchange is a pre-requisite of properly handling multi-currency fixed income
derivatives, so the :class:`~rateslib.fx.FXRates` and :class:`~rateslib.fx.FXForwards`
classes exist to allow full flexibility and expressing quantities in
consistent currencies.

Additionally *rateslib* also includes certain *FX Option*
products and the ability to
construct an :class:`~rateslib.fx_volatility.FXDeltaVolSmile` or
:class:`~rateslib.fx_volatility.FXDeltaVolSurface` and
:class:`~rateslib.fx_volatility.FXSabrSmile` or :class:`~rateslib.fx_volatility.FXSabrSurface`
for pricing.

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<fx-doc>`

.. raw:: html

    </div>

Can ``Curves`` be constructed and plotted in *rateslib*?
===========================================================

**Of course**. Building curves is a necessity for pricing fixed income instruments.
*Rateslib* has two primitive curve structures; :class:`~rateslib.curves.Curve`, which
is **discount factor, or survival probability based** and which can also calculate index values
for use with inflation products, for example, and :class:`~rateslib.curves.LineCurve`
(which is **purely value based**). All *Curve* types offer
various interpolation methods, such as log-linear or log-cubic spline and can even splice certain
interpolation types together.

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<curves-doc>`

.. raw:: html

    </div>

Does *rateslib* ``solve`` curves relative to market prices?
===========================================================

**Yes**, when a :class:`~rateslib.solver.Solver` is configured along with all the intended
*Instruments* and their relevant *prices*.
Multiple algorithms (*gradient descent, Gauss-Newton, Levenberg-Marquardt*) and stopping criteria
can be used within the optimization routine
to simultaneously solve multiple *Curve* parameters.

The *Solver* can even construct dependency chains, like sequentially building curves
with dependencies to other desks on an investment bank trading floor, and internally manage all of
the **risk sensitivity** calculations.

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<c-solver-doc>`

.. raw:: html

    </div>

Does *rateslib* use ``automatic differentiation (AD)``?
===========================================================

**Yes** fully integrated into all calculations.
The *dual* module provides *rateslib* with its own automatic differentiation toolset,
primarily the dual datatypes :class:`~rateslib.dual.Dual` and
:class:`~rateslib.dual.Dual2`, which operate in forward mode
(as opposed to backwards, or adjoint, mode). This allows native calculations to store first
(or second) derivative information as those calculations are made on-the-fly.

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<dual-doc>`

.. raw:: html

    </div>


Imports and Defaults
--------------------

*Rateslib* classes and methods are publicly exposed meaning anything can
be imported and used from the top level.

.. code-block::

   from rateslib import Curve, IRS, FXRates  # or * to blanket import everything

It is also possible to import the library as object and call objects from that,

.. code-block::

   import rateslib as rl
   curve = rl.Curve(...)

**Two** objects that are used **globally** are ``defaults`` and ``fixings``.

The ``defaults`` object from *rateslib* sets
parameters and settings that are used when otherwise not set by the user.
This object can only be imported, and changed, from the top level.

.. code-block::

   from rateslib import defaults
   defaults.base_currency = "eur"

.. code-block::

   import rateslib as rl
   rl.defaults.base_currency = "eur"

The ``fixings`` object allows a user to populate and load published fixing data from any
backend data source.

.. code-block::

   from rateslib import fixings
   fixings.add("SOME_FIXING_DATA", index=[dt(2000, 1, 1)], data=[99.66])

How to Use Rateslib
-------------------

The best way to learn *rateslib* is to follow the
tutorials and examples in the :ref:`User Guide<guide-doc>`.
This systematically introduces the main objects and concepts.
