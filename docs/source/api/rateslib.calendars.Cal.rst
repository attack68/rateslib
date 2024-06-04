Cal
==========

.. currentmodule:: rateslib.calendars

.. py:class:: Cal(holidays, weel_mask)

   A calendar object for making date roll adjustment calculations.

   :param holidays: A list of dates that defines holidays.
   :type holidays: list of datetime

   :param week_mask: A list of days defined as non-working days of the week. Most common is `[5, 6]` for Sat and Sun.
   :type vars: list of int

   .. rubric:: Attributes

   :ivar holidays: list of datetime
   :ivar week_mask: list of int

   .. seealso::
      :class:`~rateslib.calendars.UnionCal`: Calendar object designed to merge calendars under financial date rules.

   .. rubric:: Examples

   .. ipython:: python

      from rateslib.dual import Dual, gradient
      def func(x, y):
          return 5 * x**2 + 10 * y**3

      x = Dual(1.0, ["x"], [])
      y = Dual(1.0, ["y"], [])
      gradient(func(x,y), ["x", "y"])

   .. rubric:: Methods Summary

   .. include:: rateslib.calendars.Cal.add_days.rst
   .. include:: rateslib.calendars.Cal.add_bus_days.rst
   .. include:: rateslib.calendars.Cal.bus_date_range.rst
   .. include:: rateslib.calendars.Cal.is_bus_day.rst
   .. include:: rateslib.calendars.Cal.is_non_bus_day.rst
   .. include:: rateslib.calendars.Cal.lag.rst
   .. include:: rateslib.calendars.Cal.roll.rst


