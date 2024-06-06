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

   from rateslib.calendars.rs import CALENDARS
   print(CALENDARS.keys())

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


Day count fractions (DCFs)
**************************

This module also contains a :meth:`~rateslib.calendars.dcf` method for calculating
day count fractions.
Review the API documentation for specific calculation details. Current DCF conventions
available are listed below:

.. ipython:: python

   from rateslib.calendars import _DCF
   print(_DCF.keys())
