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

   .. rubric:: Methods Summary

   .. autofunction:: rateslib.rs.Cal.add_days
   .. autofunction:: rateslib.rs.Cal.add_bus_days
   .. include:: rateslib.calendars.Cal.add_months.rst
   .. include:: rateslib.calendars.Cal.bus_date_range.rst
   .. include:: rateslib.calendars.Cal.cal_date_range.rst
   .. autofunction:: rateslib.rs.Cal.is_bus_day
   .. autofunction:: rateslib.rs.Cal.is_non_bus_day
   .. autofunction:: rateslib.rs.Cal.is_settlement
   .. include:: rateslib.calendars.Cal.lag.rst
   .. include:: rateslib.calendars.Cal.roll.rst
   .. include:: rateslib.calendars.Cal.to_json.rst


