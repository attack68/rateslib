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

Defaults
---------

.. automodapi:: rateslib.default
   :no-heading:
   :no-inheritance-diagram:
   :skip: plot
   :skip: BusinessDay

Calendars
---------

.. automodapi:: rateslib.calendars
   :no-heading:
   :skip: floor
   :skip: next_monday
   :skip: next_monday_or_tuesday

Scheduling
----------

.. automodapi:: rateslib.scheduling
   :no-heading:
   :no-inheritance-diagram:
   :skip: Any
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Day
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
   :skip: PPSpline
   :skip: datetime
   :skip: timedelta
   :skip: Holiday
   :skip: comb


FX
---

.. automodapi:: rateslib.fx
   :no-heading:
   :skip: Any
   :skip: Curve
   :skip: LineCurve
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Dual
   :skip: Series
   :skip: datetime
   :skip: product
   :skip: timedelta

Periods
-------

.. automodapi:: rateslib.periods
   :no-heading:
   :skip: ABCMeta
   :skip: IndexCurve
   :skip: Curve
   :skip: CustomBusinessDay
   :skip: DataFrame
   :skip: Dual
   :skip: Dual2
   :skip: FXRates
   :skip: FXForwards
   :skip: LineCurve
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
   :skip: forward_fx
   :skip: sqrt
   :skip: abstractmethod
   :skip: add_tenor
   :skip: concat
   :skip: date_range
   :skip: dcf
   :skip: get_calendar
   :skip: index_left
   :skip: set_order
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
   :skip: CustomLeg
   :skip: FixedLeg
   :skip: FixedLegExchange
   :skip: FixedLegExchangeMtm
   :skip: FloatLeg
   :skip: FloatLegExchange
   :skip: FloatLegExchangeMtm
   :skip: ZeroFloatLeg
   :skip: LineCurve
   :skip: Solver
   :skip: CompositeCurve
   :skip: ZeroFixedLeg

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