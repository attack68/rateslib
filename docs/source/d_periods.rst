.. _periods-doc:

.. ipython:: python
   :suppress:

   from rateslib.periods import *
   from rateslib.curves import *
   from datetime import datetime as dt

***********
Periods
***********

The ``rateslib.periods`` module creates *Period* objects that define ways to
describe single cashflows,
generated under various calculation methodologies. It is probably quite rare that
*Periods* will be instantiated directly, rather they form the components of
:ref:`Legs<legs-doc>`, but none-the-less, this page describes their construction.

The following *Periods* are provided, click on the links for a full description
of each *Period* type:

.. autosummary::
   rateslib.periods.BasePeriod
   rateslib.periods.FixedPeriod
   rateslib.periods.FloatPeriod
   rateslib.periods.Cashflow
   rateslib.periods.IndexFixedPeriod
   rateslib.periods.IndexCashflow

Every *Period* type is endowed with the following the methods:

.. autosummary::
   rateslib.periods.BasePeriod.npv
   rateslib.periods.BasePeriod.analytic_delta
   rateslib.periods.BasePeriod.cashflows

:class:`~rateslib.periods.FloatPeriod` types have specific methods to support
their specific functionality, such as:

.. autosummary::
   rateslib.periods.FloatPeriod.rate
   rateslib.periods.FloatPeriod.fixings_table

:class:`~rateslib.periods.IndexFixedPeriod` and
:class:`~rateslib.periods.IndexCashflow` types have specific methods to support
their specific functionality, such as:

.. autosummary::
   rateslib.periods.IndexMixin.index_ratio

.. .. autoclass:: rateslib.periods.BasePeriod
      :members:
   .. autoclass:: rateslib.periods.FixedPeriod
   .. autoclass:: rateslib.periods.FloatPeriod
      :members: rate, fixings_table
   .. autoclass:: rateslib.periods.Cashflow