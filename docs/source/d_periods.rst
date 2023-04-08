.. _periods-doc:

.. ipython:: python
   :suppress:

   from rateslib.periods import *
   from rateslib.curves import *
   from datetime import datetime as dt

***********
Periods
***********

The ``rateslib.periods`` module creates ``Period`` objects that define ways to
describe single cashflows,
generated under various calculation methodologies. It is probably quite rare that
``Periods`` will be instantiated directly, rather they form the components of
:ref:`Legs<legs-doc>`, but none-the-less, this page describes their construction.
The following ``Periods`` are provided:

.. autosummary::
   rateslib.periods.BasePeriod
   rateslib.periods.FixedPeriod
   rateslib.periods.FloatPeriod
   rateslib.periods.Cashflow

Every ``Period`` type is endowed with the following the methods:

.. autosummary::
   rateslib.periods.BasePeriod.npv
   rateslib.periods.BasePeriod.analytic_delta
   rateslib.periods.BasePeriod.cashflows

:class:`~rateslib.periods.FloatPeriod` types have specific methods to support
their specific functionality, such as:

.. autosummary::
   rateslib.periods.FloatPeriod.rate
   rateslib.periods.FloatPeriod.fixings_table

Fixed Periods
-------------

A :class:`~rateslib.periods.FixedPeriod` is a simpler object
than :class:`~rateslib.periods.FloatPeriod` since the
cashflow that it defines is explicit and does not depend upon a curve and a rate
calculation. The ``start`` and ``end`` arguments define the accrual period, through
which the DCF calculation will be determined along with the ``convention`` argument
(and if necessary the ``termination``, ``frequency`` and ``stub`` arguments). The
``notional`` and ``fixed_rate`` will determine the cashflow size and direction.
When the ``payment`` date
does not align with the accrual ``end`` date, which is common for many derivatives
with lagged payment schedules this allows for direct specification.

.. note::

   A positive ``notional`` is interpreted in ``rateslib`` as **paying** a ``Period``
   or a ``Leg``.
   This means that the associated cashflow for that period is negative.

.. ipython:: python

   fixed_period = FixedPeriod(
       start=dt(2022,1,1),
       end=dt(2022,7,1),
       payment=dt(2022,7,2),
       frequency="S",
       notional=1000000,
       convention="Act365F",
       fixed_rate=2.10,
   )
   fixed_period.cashflows()

A 6 month period with a 1mm notional typically has an ``analytic_delta`` of around
50 local currency units per bp (basis point), dependent upon the level of rates on the
``curve``.

.. ipython:: python

   fixed_period.analytic_delta(curve)

Cashflow
--------

:class:`~rateslib.periods.Cashflow` allows fixed payment amounts to be similarly
defined explicitly,
as a specific ``notional`` amount on a ``payment`` date with **no other dependencies**.
For this reason its ``analytic_delta`` is zero.
This definition allows the ``analytic_delta``
of composited legs, which might contain both :class:`~rateslib.periods.Cashflow` and
:class:`~rateslib.periods.FixedPeriod` s to correctly identify a sensitivity
to the change in fixed rate, which would not impact the notional cashflows.

.. ipython:: python

   custom_period = Cashflow(
      notional=10413.70,
      payment=dt(2022,7,2)
   )
   custom_period.cashflows(curve)
   custom_period.npv(curve)
   custom_period.analytic_delta(curve)


Floating Periods
----------------

A :class:`~rateslib.periods.FloatPeriod` uses the same kind of construction
as a :class:`~rateslib.periods.FixedPeriod` , except
that, to calculate its cashflow, a :class:`~rateslib.curves.Curve` or
:class:`~rateslib.curves.LineCurve` and a method for determining the
:meth:`~rateslib.periods.FloatPeriod.rate` is required.

For example,

.. ipython:: python

   curve = Curve(
       nodes={dt(2021,1,1): 1.00, dt(2025,1,1): 0.83},
       interpolation="log_linear",
       id="sonia"
   )
   float_period = FloatPeriod(
       start=dt(2021,1,1),
       end=dt(2021,7,1),
       payment=dt(2021,7,2),
       frequency="S",
       notional=1000000,
       currency="gbp",
       convention="Act360",
       fixing_method="rfr_payment_delay",
   )
   float_period.cashflows(curve, fx=1.25)
   float_period.npv(curve)
   float_period.analytic_delta(curve)


.. .. autoclass:: rateslib.periods.BasePeriod
      :members:
   .. autoclass:: rateslib.periods.FixedPeriod
   .. autoclass:: rateslib.periods.FloatPeriod
      :members: rate, fixings_table
   .. autoclass:: rateslib.periods.Cashflow