.. _api-doc:

**************
API Reference
**************

.. Global imports for iPython are executed in the doc string for add_tenor which is the
   first function indexed alphabetically by automodapi.

Indices and tables
===================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Notation
=========

.. math::

   d \quad \text{or} \quad d_i =& \text{DCF of period} \; (i) \\
   m \quad \text{or} \quad m_i =& \text{Maturity date of period} \; (i) \\
   v(m) =& \text{DF of period payment date,} \; m \\
   N \quad \text{or} \quad N_i =& \text{Notional of period} \; (i) \\
   R =& \text{Fixed rate of period or leg} \\
   z =& \text{Floating period spread} \\
   r(r_i, z) =& \text{Floating rate of period as a function of fixings,} \; (r_i) \\
   C =& \text{Cashflow} \\
   P =& \text{Net present value} \\
   I(m) =& \text{Index ratio applicable at maturity,} \; m \\

Defaults
=========

.. automodapi:: rateslib.default
   :no-heading:
   :no-inheritance-diagram:
   :skip: plot
   :skip: plot3d
   :skip: datetime
   :skip: Enum
   :skip: read_csv
   :skip: get_named_calendar

Calendars
==========

Classes
^^^^^^^

.. autosummary::
   ~rateslib.calendars.Cal
   ~rateslib.calendars.NamedCal
   ~rateslib.calendars.UnionCal
   ~rateslib.calendars.Modifier
   ~rateslib.calendars.RollDay

.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    api/rateslib.calendars.Cal.rst
    api/rateslib.calendars.NamedCal.rst
    api/rateslib.calendars.UnionCal.rst
    api/rateslib.calendars.Modifier.rst
    api/rateslib.calendars.RollDay.rst

.. automodapi:: rateslib.calendars
   :no-heading:

Scheduling
===========

.. automodapi:: rateslib.scheduling
   :no-heading:
   :no-inheritance-diagram:
   :skip: Any
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Iterator
   :skip: datetime
   :skip: product
   :skip: timedelta
   :skip: NoInput

Highlighted private functions
-----------------------------

.. autosummary::
   _check_regular_swap
   _infer_stub_date

.. toctree::
    :hidden:

    api/rateslib.scheduling._check_regular_swap.rst
    api/rateslib.scheduling._infer_stub_date.rst

Piecewise Polynomial Splines
=============================

Functions
^^^^^^^^^

.. autosummary::
   ~rateslib.splines.bsplev_single
   ~rateslib.splines.bspldnev_single
   ~rateslib.splines.evaluate

Classes
^^^^^^^^

.. autosummary::
   ~rateslib.splines.PPSplineF64
   ~rateslib.splines.PPSplineDual
   ~rateslib.splines.PPSplineDual2


.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    api/rateslib.splines.PPSplineF64.rst
    api/rateslib.splines.PPSplineDual.rst
    api/rateslib.splines.PPSplineDual2.rst
    api/rateslib.splines.bsplev_single.rst
    api/rateslib.splines.bspldnev_single.rst
    api/rateslib.splines.evaluate.rst


Dual (for AD)
==============

.. automodapi:: rateslib.dual
   :no-heading:

Classes
^^^^^^^^

.. autosummary::
   Dual
   Dual2

.. toctree::
    :maxdepth: 0
    :titlesonly:
    :hidden:

    api/rateslib.dual.Dual.rst
    api/rateslib.dual.Dual2.rst


Curves
=======

.. automodapi:: rateslib.curves
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Class Inheritance Diagram
--------------------------

.. automod-diagram:: rateslib.curves
   :parts: 1

FX
===

.. automodapi:: rateslib.fx
   :no-heading:
   :no-inheritance-diagram:

FX Volatility
==============

.. automodapi:: rateslib.fx_volatility
   :no-heading:
   :skip: set_order_convert
   :skip: dual_exp
   :skip: dual_inv_norm_cdf
   :skip: DualTypes
   :skip: Dual
   :skip: Dual2
   :skip: dual_norm_cdf
   :skip: dual_log
   :skip: dual_norm_pdf
   :skip: PPSplineF64
   :skip: PPSplineDual
   :skip: PPSplineDual2
   :skip: evaluate
   :skip: plot
   :skip: NoInput
   :skip: newton_1dim
   :skip: uuid4
   :skip: Union
   :skip: datetime
   :skip: dt
   :skip: timedelta
   :skip: Series

Periods
========

Link to the :ref:`Periods<periods-doc>` section in the user guide.

.. automodapi:: rateslib.periods
   :no-heading:
   :skip: NoInput
   :skip: ABCMeta
   :skip: IndexCurve
   :skip: Curve
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXRates
   :skip: FXForwards
   :skip: LineCurve
   :skip: CompositeCurve
   :skip: Series
   :skip: datetime
   :skip: comb
   :skip: FXDeltaVolSurface
   :skip: FXDeltaVolSmile

Legs
=====

Link to the :ref:`Legs<legs-doc>` section in the user guide.

.. automodapi:: rateslib.legs
   :no-heading:

Instruments
============

.. automodapi:: rateslib.instruments
   :no-heading:
   :inherited-members:
   :no-inheritance-diagram:

Class Inheritance Diagram
--------------------------

.. automod-diagram:: rateslib.instruments
    :parts: 1

Solver
=======

.. automodapi:: rateslib.solver
   :no-heading:
   :skip: MultiCsaCurve
   :skip: NoInput
   :skip: FXRates
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXForwards
   :skip: combinations
   :skip: MultiIndex
   :skip: Series
   :skip: CompositeCurve
   :skip: ProxyCurve
   :skip: concat
   :skip: dual_log
   :skip: dual_solve
   :skip: gradient
   :skip: time
   :skip: uuid4
   :skip: PerformanceWarning

Cookbook
=========

**Curve Building**

.. toctree::
    :titlesonly:

    z_swpm.rst
    z_dependencychain.rst
    z_turns.rst
    z_quantlib.rst
    z_curve_from_zero_rates.ipynb

**FX Volatility Surface Building**

.. toctree::
    :titlesonly:

    z_eurusd_surface.ipynb
    z_fxvol_temporal.ipynb

**Instrument Pricing**

.. toctree::
    :titlesonly:

    z_stubs.rst
    z_fixings.rst
    z_historical_swap.ipynb
    z_amortization.rst
    z_reverse_xcs.rst

**Risk Sensitivity Analysis**

.. toctree::
    :titlesonly:

    z_convexityrisk.rst
    z_bondbasis.rst
    z_bondctd.rst
