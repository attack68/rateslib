NamedCal
==========

.. currentmodule:: rateslib.calendars

.. py:class:: NamedCal(name)

   A calendar wrapper of a :class:`~rateslib.calendars.UnionCal` created by a string name.

   See :ref:`User Guide: Calendars <settlement-cals>`.

   :param name: A string defining the calendar. Named calendars separated by commas, and associated settlement calendars separated by pipes. Valid examples are; "tgt", "tgt,ldn", "tgt,ldn|fed,stk",
   :type calendars: str

   .. rubric:: Attributes Documentation

   .. autoattribute:: name
   .. autoattribute:: union_cal

   .. seealso::
      :class:`~rateslib.calendars.Cal`: Base calendar object type.
      :class:`~rateslib.calendars.UnionCal`: Calendar object designed to merge calendars under financial date rules.

   .. rubric:: Methods Summary

   .. autosummary::

      ~NamedCal.add_bus_days
      ~NamedCal.add_cal_days
      ~NamedCal.add_months
      ~NamedCal.bus_date_range
      ~NamedCal.cal_date_range
      ~NamedCal.is_bus_day
      ~NamedCal.is_non_bus_day
      ~NamedCal.is_settlement
      ~NamedCal.lag_bus_days
      ~NamedCal.adjust
      ~NamedCal.adjusts
      ~NamedCal.roll
      ~NamedCal.to_json

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
