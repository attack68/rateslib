.. roll
   ==========

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.roll(date, modifier, settlement)

   Adjust a non-business date to a business date under a specific modification rule.

   :param date: The date to adjust.
   :type date: datetime

   :param modifier: The modification rule.
   :type modifier: Modifier

   :rtype: datetime

   .. rubric:: Notes

   An input business date will be returned unchanged.