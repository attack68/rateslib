.. add_days
   ==========

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.add_days(date, days, modifier, settlement)

   Return a date separated by calendar days from input date, and rolled with a modifier.

   :param date: The original business date. Raise if a non-business date is given.
   :type date: datetime

   :param days: The number of calendar days to add.
   :type days: int

   :param modifier: The rule to use to roll resultant non-business days.
   :type modifier: Modifier

   :param settlement: Enforce an associated settlement calendar, if *True* and if one exists.
   :type settlement: bool

   :rtype: datetime
