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

Or, it can be installed via the community ``conda-forge`` channel available from
`Anaconda.org <https://anaconda.org/conda-forge/rateslib>`_

.. code-block::

   conda install --channel=conda-forge rateslib

**Minimum Dependencies**

.. list-table::
   :widths: 20 20 20 40
   :header-rows: 1


   * - Package Name
     - Recommended Min Version
     - Earliest Tested
     - Comment
   * - Python
     - 3.11
     - 3.9
     -
   * - NumPy
     - 1.23.5
     - 1.21.5
     -
   * - Pandas
     - 1.5.3
     - 1.4.1
     - (2.0 is currently untested)
   * - Matplotlib
     - 3.6.3
     - 3.5.1
     - (used for plotting curves)


Introduction to Rateslib
-------------------------

.. raw:: html

    <div class="container">
    <div id="accordion" class="shadow tutorial-accordion">

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseOne">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Which &nbsp;<span style="color: darkorange;">fixed income instruments</span>&nbsp; does rateslib include?
                    </div>
                    <span class="badge gs-badge-link">

:ref:`Straight to tutorial...<instruments-doc>`

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseOne" class="collapse" data-parent="#accordion">
                <div class="card-body">

**Securities**

- Discount securities - :class:`~rateslib.instruments.Bill`,
- Regular nominal bonds - :class:`~rateslib.instruments.FixedRateBond`,
- Bond futures - :class:`~rateslib.instruments.BondFuture`,
- Index linked bonds - :class:`~rateslib.instruments.IndexFixedRateBond`,
- Also FRNs - :class:`~rateslib.instruments.FloatRateBond`.

**Single Currency Derivatives**

- Interest rate swaps (both IBOR and RFR) - :class:`~rateslib.instruments.IRS`,
- Index interest rate swaps - :class:`~rateslib.instruments.IIRS`,
- Zero coupon swaps - :class:`~rateslib.instruments.ZCS`,
- Zero coupon index swaps - :class:`~rateslib.instruments.ZCIS`,
- Basis swaps - :class:`~rateslib.instruments.SBS`,
- Forward rate agreements - :class:`~rateslib.instruments.FRA`.

**Multi-currency Derivatives**

- Cross-currency basis swaps - :class:`~rateslib.instruments.XCS`,
- Non-mtm cross-currency basis swaps - :class:`~rateslib.instruments.NonMtmXCS`,
- FX swaps - :class:`~rateslib.instruments.FXSwap`.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseAD">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Does rateslib use &nbsp;<span style="color: darkorange;">automatic differentiation (AD)</span>?
                    </div>
                    <span class="badge gs-badge-link">

:ref:`Straight to tutorial...<dual-doc>`

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseAD" class="collapse" data-parent="#accordion">
                <div class="card-body">

**Yes**. The ``rateslib.dual`` module provides ``rateslib`` with its own integrated
automatic differentiation toolset, using dual numbers, which operate in forward mode
(as opposed to backwards, or adjoint, mode).

Whenever you see a calculation result that displays a ``<Dual: 2.40..>`` datatype,
it is sufficient to understand that the real value associated with this is the
answer (2.40), and can be extracted directly with ``float(result)`` or ``result.real``.
The ``dual`` attribute of the result contains first derivative information with regards
to variables that has attributed to its calculation.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseTwo">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Does rateslib handle &nbsp;<span style="color: darkorange;">foreign exchange (FX)</span>?
                    </div>
                    <span class="badge gs-badge-link">

:ref:`Straight to tutorial...<fx-doc>`

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseTwo" class="collapse" data-parent="#accordion">
                <div class="card-body">

**Yes**. Foreign exchange is a pre-requisite of properly handling multi-currency
derivatives, so the :class:`~rateslib.fx.FXRates` and :class:`~rateslib.fx.FXForwards`
classes exist to allow full flexibility and expressing quantities in
consistent currencies.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseThree">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Can rateslib create and plot &nbsp;<span style="color: darkorange;">interest rate curves</span>?
                    </div>
                    <span class="badge gs-badge-link">

:ref:`Straight to tutorial...<curves-doc>`

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseThree" class="collapse" data-parent="#accordion">
                <div class="card-body">

**Of course**. Building curves is a necessity for pricing fixed income instruments.
``rateslib`` has two available curve structures, and within those different
interpolation options:

- :class:`~rateslib.curves.Curve`, which is **discount factor based**. The native
  interpolation options provided for these are ``log_linear``, ``linear_zero_rate``,
  ``log-cubic spline``, or
  a mixture of the two with the longer end being log-cubic spline.
- :class:`~rateslib.curves.LineCurve`, which is **purely value based**. The native
  interpolation options provided for these are ``linear``, ``flat_forward``,
  ``cubic spline``, or a
  mixture of the two with the longer end being cubic spline.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

        <div class="card tutorial-card">
            <div class="card-header collapsed card-link" data-toggle="collapse" data-target="#collapseFour">
                <div class="d-flex flex-row tutorial-card-header-1">
                    <div class="d-flex flex-row tutorial-card-header-2">
                        <button class="btn btn-dark btn-sm"></button>
                        Can rateslib &nbsp;<span style="color: darkorange;">solve</span>&nbsp; interest rates curves from market instruments?
                    </div>
                    <span class="badge gs-badge-link">

:ref:`Straight to tutorial...<c-solver-doc>`

.. raw:: html

                    </span>
                </div>
            </div>
            <div id="collapseFour" class="collapse" data-parent="#accordion">
                <div class="card-body">

**Absolutely**. ``rateslib`` has a state-of-the-art  :class:`~rateslib.solver.Solver`,
which can use multiple algorithms (*gradient descent, Gauss-Newton, Leveberg-Marquardt*)
to simultaneously solve the curve parameters to fit provided market instrument prices.

The solver can even construct dependency chains, like sequentially building curves
with dependencies to other desks in on an investment bank trading floor, and it can
handle over-specified curves or under-specified curves.

.. raw:: html

                </div>
            </div>
        </div>

.. raw:: html

    </div>
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

The documentation often imports directly from the underlying code modules for greater
clarity. There is no operational difference
in any of theses importing methods, and all are valid.

.. code-block::

   from rateslib.curves import Curve
   from rateslib.fx import FXRates
   from rateslib.instruments import IRS

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
