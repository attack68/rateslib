Cal
==========

.. currentmodule:: rateslib.calendars

.. py:class:: Cal(holidays, week_mask)

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

   .. autosummary::

      ~Cal.add_bus_days
      ~Cal.add_cal_days
      ~Cal.add_months
      ~Cal.bus_date_range
      ~Cal.cal_date_range
      ~Cal.is_bus_day
      ~Cal.is_non_bus_day
      ~Cal.is_settlement
      ~Cal.lag_bus_days
      ~Cal.roll
      ~Cal.to_json

   .. rubric:: Attributes Documentation

   .. autoattribute:: holidays
   .. autoattribute:: week_mask

   .. rubric:: Methods Documentation

   .. automethod:: add_bus_days
   .. automethod:: add_cal_days
   .. automethod:: add_months
   .. automethod:: bus_date_range
   .. automethod:: cal_date_range
   .. automethod:: is_bus_day
   .. automethod:: is_non_bus_day
   .. automethod:: is_settlement
   .. automethod:: lag_bus_days
   .. automethod:: roll
   .. automethod:: to_json