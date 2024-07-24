.. _cal-doc:

.. ipython:: python
   :suppress:

   from rateslib.calendars import *
   from datetime import datetime as dt

************
Calendars
************

The ``rateslib.calendars`` module generates holiday calendars so that
business days are well defined.

Summary
*******

Classes
--------
.. autosummary::
   rateslib.calendars.Cal
   rateslib.calendars.NamedCal
   rateslib.calendars.UnionCal

Methods
-------
.. autosummary::
   rateslib.calendars.get_calendar
   rateslib.calendars.add_tenor
   rateslib.calendars.dcf

Loading existing calendars
***************************

.. warning::

   Use preset calendars at your own risk. Generally the repeated yearly holidays are
   accurate but a full list of ad-hoc and specialised holidays has not been properly
   reviewed and is not necessarily upto date.

The :meth:`~rateslib.calendars.get_calendar` method is used internally
to parse the different
options a user might provide, e.g. supplying *NoInput* and then generating a
null calendar object with no holidays or passing through a user defined
calendar. There are also some calendars
pre-programmed, and those currently available calendars are below. More information
is available in the :meth:`~rateslib.calendars.get_calendar` method:

.. ipython:: python

   from rateslib import defaults
   print(defaults.calendars.keys())

.. ipython:: python

   ldn_cal = get_calendar("ldn")
   ldn_cal.bus_date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9))
   stk_cal = get_calendar("stk")
   stk_cal.bus_date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9))

Available calendars can also be **combined** if a comma separator is used in the
argument, which acts as an AND operator for business days and an OR operator for
holidays. This is useful for multi-currency derivatives.

.. ipython:: python

   ldn_stk_cal = get_calendar("ldn,stk")
   ldn_stk_cal.bus_date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9))

Creating a custom calendar
**************************

Custom calendars are directly constructed from the :class:`~rateslib.calendars.Cal` class.
This requires a list of ``holidays`` and a ``week_mask``.

.. ipython:: python

   custom_cal = Cal([dt(2023, 12, 25), dt(2023, 12, 26), dt(2024, 1, 1)], [5, 6])
   custom_cal.bus_date_range(start=dt(2023, 12, 18), end=dt(2024, 1, 5))

.. _settlement-cals:

Calendar combinations
**********************

Custom calendar combinations can be constructed with the :class:`~rateslib.calendars.UnionCal`
class. It requires a list of *Cal* objects to form the union of non-business dates,
and another, secondary list, of associated ``settlement_calendars``, to validate
calculated dates against.

Combined calendars can also be constructed automatically with string parsing.
Comma separation forms a union of calendars. For example the appropriate calendar
for a EUR/USD cross-currency swap is *"tgt,nyc"* for TARGET and New York.

The appropriate holiday calendar to use for a EURUSD FX instrument, such as spot
determination is *"tgt|nyc"*, which performs date manipulation under a TARGET calendar
but enforces associated settlement against the New York calendar.

.. ipython:: python

   # Combined calendar with no associated settlement calendar
   tgt_nyc = get_calendar("tgt,nyc")
   tgt_nyc.is_bus_day(dt(2009, 11, 11))
   tgt_nyc.is_settlement(dt(2009, 11, 11))

   # TARGET calendar enforcing New York settlement
   tgt_nyc_settle = get_calendar("tgt|nyc")
   tgt_nyc_settle.is_bus_day(dt(2009, 11, 11))
   tgt_nyc_settle.is_settlement(dt(2009, 11, 11))


Adding Custom Calendars to Defaults
**************************************

Custom calendars can be added to the ``defaults`` object and this allows the
:meth:`~rateslib.calendars.get_calendar` method to access it via string representation
in *Instrument* instantiation or or in other methods such as :meth:`~rateslib.calendars.add_tenor`.

Suppose we create a custom calendar which allows only Wednesdays to be business days.
We can then use this calendar to derive IMM dates in a month, although this is
not the most efficient way of doing this it is just shown for example purposes.

.. ipython:: python

   cal = Cal([], [0, 1, 3, 4, 5, 6])
   defaults.calendars["wednesdays"] = cal

   # The IMM date in March 2025 is..
   add_tenor(dt(2025, 3, 15), "0d", calendar="wednesdays", modifier="F")

Whenever the ``calendar`` argument is required for a method the string *'wednesdays'* could
now be freely used and would refer back to this object.

.. ipython:: python
   :suppress:

   defaults.reset_defaults()

Day count fractions (DCFs)
**************************

This module also contains a :meth:`~rateslib.calendars.dcf` method for calculating
day count fractions.
Review the API documentation for specific calculation details. Current DCF conventions
available are listed below:

.. ipython:: python

   from rateslib.calendars import _DCF
   print(_DCF.keys())
