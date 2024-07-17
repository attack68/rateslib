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
   .. autofunction:: rateslib.rs.Cal.add_months
   .. autofunction:: rateslib.rs.Cal.bus_date_range
   .. autofunction:: rateslib.rs.Cal.cal_date_range
   .. autofunction:: rateslib.rs.Cal.is_bus_day
   .. autofunction:: rateslib.rs.Cal.is_non_bus_day
   .. autofunction:: rateslib.rs.Cal.is_settlement
   .. autofunction:: rateslib.rs.Cal.lag
   .. autofunction:: rateslib.rs.Cal.roll
   .. autofunction:: rateslib.rs.Cal.to_json.rst
