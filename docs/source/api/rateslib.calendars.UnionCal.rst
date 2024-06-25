UnionCal
==========

.. currentmodule:: rateslib.calendars

.. py:class:: UnionCal(calendars, settlement_calendars)

   A calendar object for making date roll adjustment calculations, combining multiple calendars.

   :param calendars: A list of `Cal` objects.
   :type calendars: list of Cal

   :param settlement_calendars: A list of `Cal` objects used only as associated settlement calendars.
   :type vars: list of Cal

   .. rubric:: Attributes

   :ivar holidays: list of datetime
   :ivar week_mask: list of int

   .. seealso::
      :class:`~rateslib.calendars.Cal`: Base calendar object type.

   .. rubric:: Methods Summary

   .. include:: rateslib.calendars.UnionCal.add_days.rst
   .. include:: rateslib.calendars.UnionCal.add_bus_days.rst
   .. include:: rateslib.calendars.UnionCal.add_months.rst
   .. include:: rateslib.calendars.UnionCal.bus_date_range.rst
   .. include:: rateslib.calendars.UnionCal.cal_date_range.rst
   .. include:: rateslib.calendars.UnionCal.is_bus_day.rst
   .. include:: rateslib.calendars.UnionCal.is_non_bus_day.rst
   .. include:: rateslib.calendars.UnionCal.is_settlement.rst
   .. include:: rateslib.calendars.UnionCal.lag.rst
   .. include:: rateslib.calendars.UnionCal.roll.rst
   .. include:: rateslib.calendars.UnionCal.to_json.rst


