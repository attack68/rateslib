.. lag
   ==========

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.lag(date, days)

   Return a business date separated by `days` from input `date`.

   :param date: The original business date. Raise if a non-business date is given.
   :type other: datetime

   :param days: Number of business days to add.
   :type days: int

   :rtype: datetime
