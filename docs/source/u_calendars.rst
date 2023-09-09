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
It is built upon the ``pandas`` holiday calendars methods, which are themselves
extensions of ``numpy`` data structures.

Summary
*******

Methods
-------
.. autosummary::
   rateslib.calendars.create_calendar
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

   from rateslib.calendars import CALENDARS
   print(CALENDARS.keys())

.. ipython:: python

   ldn_cal = get_calendar("ldn")
   date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9), freq=ldn_cal)
   stk_cal = get_calendar("stk")
   date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9), freq=stk_cal)

Available calendars can also be **combined** if a comma separator is used in the
argument, which acts as an AND operator for business days and an OR operator for
holidays. This is useful for multi-currency derivatives.

.. ipython:: python

   ldn_stk_cal = get_calendar("ldn,stk")
   date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 9), freq=ldn_stk_cal)

Creating a custom calendar
**************************

The :meth:`~rateslib.calendars.create_calendar` method is provided to allow
a user to create their
own custom calendar defined by a weekmask and specific holidays. For example,
suppose one wanted to create a holiday calendar that included weekends and
Christmas every year and New Year's Day rolled forward to a monday if it
happened to fall on a weekend. The approach is as follows,

.. ipython:: python

   from pandas import date_range
   from pandas.tseries.holiday import Holiday, next_monday
   holidays = [
       Holiday("Christmas", month=12, day=25),
       Holiday("New Year's", month=1, day=1, observance=next_monday),
   ]
   custom_cal = create_calendar(holidays, "Mon Tue Wed Thu Fri")
   date_range(start=dt(2022, 12, 23), end=dt(2023, 1, 5), freq=custom_cal)

Day count fractions (DCFs)
**************************

This module also contains a :meth:`~rateslib.calendars.dcf` method for calculating
day count fractions.
Review the API documentation for specific calculation details. Current DCF conventions
available are listed below:

.. ipython:: python

   from rateslib.calendars import _DCF
   print(_DCF.keys())
