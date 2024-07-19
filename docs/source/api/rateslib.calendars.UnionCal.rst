UnionCal
==========

.. currentmodule:: rateslib.calendars

.. py:class:: UnionCal(calendars, settlement_calendars)

   A calendar object for making date roll adjustment calculations, combining multiple calendars.

   See :ref:`User Guide: Calendars <settlement-cals>`.

   :param calendars: A list of `Cal` objects.
   :type calendars: list of Cal

   :param settlement_calendars: A list of `Cal` objects used only as associated settlement calendars.
   :type vars: list of Cal

   .. rubric:: Attributes Documentation

   .. autoattribute:: holidays
   .. autoattribute:: week_mask

   .. seealso::
      :class:`~rateslib.calendars.Cal`: Base calendar object type.

   .. rubric:: Methods Summary

   .. autosummary::

      ~UnionCal.add_bus_days
      ~UnionCal.add_days
      ~UnionCal.add_months
      ~UnionCal.bus_date_range
      ~UnionCal.cal_date_range
      ~UnionCal.is_bus_day
      ~UnionCal.is_non_bus_day
      ~UnionCal.is_settlement
      ~UnionCal.lag
      ~UnionCal.roll
      ~UnionCal.to_json

   .. rubric:: Methods Documentation

   .. automethod:: add_bus_days
   .. automethod:: add_days
   .. automethod:: add_months
   .. automethod:: bus_date_range
   .. automethod:: cal_date_range
   .. automethod:: is_bus_day
   .. automethod:: is_non_bus_day
   .. automethod:: is_settlement
   .. automethod:: lag
   .. automethod:: roll
   .. automethod:: to_json
