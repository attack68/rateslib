UnionCal
==========

.. currentmodule:: rateslib.scheduling

.. py:class:: UnionCal(calendars, settlement_calendars)

   A calendar object for making date calculations, combining multiple calendars.

   See :ref:`User Guide: Calendars <settlement-cals>`.

   :param calendars: A list of `Cal` objects.
   :type calendars: list of Cal

   :param settlement_calendars: A list of `Cal` objects used only as associated settlement calendars.
   :type vars: list of Cal

   .. rubric:: Attributes Documentation

   .. autoattribute:: holidays
   .. autoattribute:: week_mask

   .. seealso::
      :class:`~rateslib.scheduling.Cal`: Base calendar object type.

   .. rubric:: Methods Summary

   .. autosummary::

      ~UnionCal.add_bus_days
      ~UnionCal.add_cal_days
      ~UnionCal.add_months
      ~UnionCal.bus_date_range
      ~UnionCal.cal_date_range
      ~UnionCal.is_bus_day
      ~UnionCal.is_non_bus_day
      ~UnionCal.is_settlement
      ~UnionCal.lag_bus_days
      ~UnionCal.adjust
      ~UnionCal.adjusts
      ~UnionCal.roll
      ~UnionCal.to_json

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
   .. automethod:: adjust
   .. automethod:: adjusts
   .. automethod:: roll
   .. automethod:: to_json
