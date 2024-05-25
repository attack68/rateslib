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
can be compiled locally.

**Additionally**, for versions less than 1.2.0, it can be installed via the community ``conda-forge`` channel
available from `Anaconda.org <https://anaconda.org/conda-forge/rateslib>`_

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
     - 3.11
     - 3.11
     - 3.9
   * - NumPy
     - 1.26.4
     - 1.26.1
     - 1.21.5
   * - Pandas
     - 2.2.1
     - 2.1.3
     - 1.4.1
   * - Matplotlib
     - 3.8.3
     - 3.8.1
     - 3.5.1


Introduction to Rateslib
-------------------------

For what purpose would I use *rateslib*?
=============================================

- If you want to integrate linear fixed income, FX and FX volatility analysis into your workflow with Python.
- If you desire a pain free setup process, a user-oriented API, and extensive documentation.
- If you are new to fixed income and currencies and interested to learn about basic and advanced concepts with
  tools to explore the nuances of these markets, as a companion to various authored books.

Which ``fixed income instruments`` does *rateslib* include?
===========================================================

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1


   * - Single Ccy Derivatives
     - Multi-Ccy Derivatives
     - Securities
     - FX Volatility
     - Combinations
   * - :class:`~rateslib.instruments.IRS`
     - :class:`~rateslib.instruments.FXExchange`
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
     -
     - :class:`~rateslib.instruments.BondFuture`
     - :class:`~rateslib.instruments.FXStraddle`
     -
   * - :class:`~rateslib.instruments.ZCS`
     -
     - :class:`~rateslib.instruments.IndexFixedRateBond`
     - :class:`~rateslib.instruments.FXStrangle`
     -
   * - :class:`~rateslib.instruments.ZCIS`
     -
     -
     - :class:`~rateslib.instruments.FXBrokerFly`
     -
   * - :class:`~rateslib.instruments.IIRS`
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
construct an :class:`~rateslib.fx_volatility.FXDeltaVolSmile` and
:class:`~rateslib.fx_volatility.FXDeltaVolSurface` for pricing.

.. raw:: html

    <div class="tutorial">

:ref:`Straight to tutorial...<fx-doc>`

.. raw:: html

    </div>

Can ``Curves`` be constructed and plotted in *rateslib*?
===========================================================

**Of course**. Building curves is a necessity for pricing fixed income instruments.
*Rateslib* has three primitive curve structures; :class:`~rateslib.curves.Curve` (which
is **discount factor based**), :class:`~rateslib.curves.LineCurve` (which is **purely value
based**), and :class:`~rateslib.curves.IndexCurve` (which is based on a *Curve* but also
calculates index values which is useful for inflation, for example). All *Curve* types offer
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

**Yes**. The *dual* module provides *rateslib* with its own integrated
automatic differentiation toolset, primarily the dual datatypes :class:`~rateslib.dual.Dual` and
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

The ``defaults`` object from *rateslib* sets
parameters and settings that are used when otherwise not set by the user.
This object can only be imported, and changed, from the top level.

.. code-block::

   from rateslib import defaults
   defaults.base_currency = "eur"

.. code-block::

   import rateslib as rl
   rl.defaults.base_currency = "eur"

How to Use Rateslib
-------------------

The best way to learn *rateslib* is to follow the
tutorials and examples in the :ref:`User Guide<guide-doc>`.
This systematically introduces the main objects and concepts.
