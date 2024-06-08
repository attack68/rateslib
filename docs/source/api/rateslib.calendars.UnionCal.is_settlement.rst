.. is_settlement
   ================

.. .. currentmodule:: rateslib.calendars

.. py:method:: UnionCal.is_settlement(date)

   Return whether the `date` is a business day in an associated settlement calendar.

   If no such associated settlement calendar exists this will return *True*.

   :param date: Date to test.
   :type date: datetime

   :rtype: bool
