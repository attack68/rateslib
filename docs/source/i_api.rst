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
   :skip: datetime
   :skip: Enum

Calendars
==========

.. automodapi:: rateslib.calendars
   :no-heading:
   :skip: floor
   :skip: next_monday
   :skip: next_monday_or_tuesday
   :skip: nearest_workday
   :skip: sunday_to_monday

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
   rateslib.scheduling._check_regular_swap
   rateslib.scheduling._infer_stub_date

.. toctree::
    :hidden:

    api/rateslib.scheduling._check_regular_swap.rst
    api/rateslib.scheduling._infer_stub_date.rst

Piecewise Polynomial Splines
=============================

.. automodapi:: rateslib.splines
   :no-heading:
   :no-inheritance-diagram:

.. Functions
   ^^^^^^^^^^
   .. autosummary::
      ~rateslib.splines.bsplev_single
      ~rateslib.splines.bspldnev_single


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
    api/rateslib.splines.bspldnev_single.rst
    api/rateslib.splines.bsplev_single.rst


Dual (for AD)
==============

.. automodapi:: rateslib.dual
   :no-heading:

Classes
^^^^^^^^

.. autosummary::
   ~rateslib.dual.Dual
   ~rateslib.dual.Dual2

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
   :skip: NoInput
   :skip: set_order_convert
   :skip: add_tenor
   :skip: create_calendar
   :skip: dcf
   :skip: dual_exp
   :skip: dual_log
   :skip: floor
   :skip: get_calendar
   :skip: plot
   :skip: uuid4
   :skip: Any
   :skip: CustomBusinessDay
   :skip: Dual
   :skip: Dual2
   :skip: PPSplineF64
   :skip: PPSplineDual
   :skip: PPSplineDual2
   :skip: datetime
   :skip: timedelta
   :skip: Holiday
   :skip: comb
   :skip: index_left_f64

Class Inheritance Diagram
--------------------------

.. automod-diagram:: rateslib.curves
   :parts: 1

FX
===

.. automodapi:: rateslib.fx
   :no-heading:
   :no-inheritance-diagram:
   :skip: gradient
   :skip: NoInput
   :skip: Curve
   :skip: LineCurve
   :skip: ProxyCurve
   :skip: MultiCsaCurve
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Dual
   :skip: Series
   :skip: datetime
   :skip: product
   :skip: timedelta
   :skip: add_tenor
   :skip: dual_solve
   :skip: plot
   :skip: set_order
   :skip: gradient

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
   :skip: DataFrame

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
   :skip: NoInput
   :skip: ABCMeta
   :skip: Curve
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXRates
   :skip: FXForwards
   :skip: Series
   :skip: datetime
   :skip: Cashflow
   :skip: FixedPeriod
   :skip: FloatPeriod
   :skip: Schedule
   :skip: IndexCashflow
   :skip: IndexFixedPeriod
   :skip: IndexCurve
   :skip: IndexMixin

Instruments
============

.. automodapi:: rateslib.instruments
   :no-heading:
   :inherited-members:
   :skip: NoInput
   :skip: IndexCurve
   :skip: IndexFixedLeg
   :skip: IndexMixin
   :skip: ZeroIndexLeg
   :skip: forward_fx
   :skip: abstractmethod
   :skip: add_tenor
   :skip: concat
   :skip: dcf
   :skip: get_calendar
   :skip: index_left
   :skip: ABCMeta
   :skip: Curve
   :skip: CustomBusinessDay
   :skip: MultiIndex
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXRates
   :skip: FXForwards
   :skip: Series
   :skip: datetime
   :skip: Cashflow
   :skip: FloatPeriod
   :skip: FixedLeg
   :skip: FixedLegMtm
   :skip: FloatLeg
   :skip: FloatLegMtm
   :skip: ZeroFloatLeg
   :skip: LineCurve
   :skip: Solver
   :skip: ZeroFixedLeg
   :skip: forward_fx
   :skip: partial
   :skip: timedelta
   :skip: FXCallPeriod
   :skip: FXDeltaVolSmile
   :skip: FXPutPeriod

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

Cookbook
=========

.. toctree::
    :titlesonly:

    z_swpm.rst
    z_dependencychain.rst
    z_turns.rst
    z_stubs.rst
    z_convexityrisk.rst
    z_bondbasis.rst
    z_bondctd.rst
    z_fixings.rst
    z_quantlib.rst

