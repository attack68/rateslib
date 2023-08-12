.. _api-doc:

**************
API Reference
**************

.. Global imports for iPython are executed in the doc string for add_tenor which is the
   first function indexed alphabetically by automodapi.

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Notation
--------

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
---------

.. automodapi:: rateslib.default
   :no-heading:
   :no-inheritance-diagram:
   :skip: plot
   :skip: BusinessDay
   :skip: datetime

Calendars
---------

.. automodapi:: rateslib.calendars
   :no-heading:
   :skip: floor
   :skip: next_monday
   :skip: next_monday_or_tuesday
   :skip: nearest_workday
   :skip: sunday_to_monday

Scheduling
----------

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

Highlighted private functions
*****************************

.. autosummary::
   rateslib.scheduling._check_regular_swap
   rateslib.scheduling._infer_stub_date

.. toctree::
    :hidden:

    api/rateslib.scheduling._check_regular_swap.rst
    api/rateslib.scheduling._infer_stub_date.rst

Piecewise Polynomial Splines
----------------------------
.. automodapi:: rateslib.splines
   :no-heading:
   :no-inheritance-diagram:
   :skip: dual_solve
   :skip: timedelta

Dual (for AD)
--------------
.. automodapi:: rateslib.dual
   :no-heading:
   :skip: isclose
   :skip: abstractmethod
   :skip: ABCMeta

Curves
------

.. automodapi:: rateslib.curves
   :no-heading:
   :inherited-members:
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
   :skip: PPSpline
   :skip: datetime
   :skip: timedelta
   :skip: Holiday
   :skip: comb


FX
---

.. automodapi:: rateslib.fx
   :no-heading:
   :skip: Curve
   :skip: LineCurve
   :skip: ProxyCurve
   :skip: CompositeCurve
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

Periods
-------

Link to the :ref:`Periods<periods-doc>` section in the user guide.

.. automodapi:: rateslib.periods
   :no-heading:
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

Legs
----

.. automodapi:: rateslib.legs
   :no-heading:
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
-----------

.. automodapi:: rateslib.instruments
   :no-heading:
   :inherited-members:
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
   :skip: MultIndex
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
   :skip: FixedLeg
   :skip: FixedLegMtm
   :skip: FloatLeg
   :skip: FloatLegMtm
   :skip: ZeroFloatLeg
   :skip: LineCurve
   :skip: Solver
   :skip: CompositeCurve
   :skip: ZeroFixedLeg
   :skip: forward_fx

Solver
------

.. automodapi:: rateslib.solver
   :no-heading:
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXForwards
   :skip: combinations
   :skip: MultiIndex
   :skip: Series
   :skip: CompositeCurve
   :skip: ProxyCurve