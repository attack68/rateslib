.. add_months
   ==========

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.add_months(date, months, modifier, roll, settlement)

   Return a date separated by months from an input date, and rolled with a modifier.

   :param date: The original date to adjust.
   :type date: datetime

   :param months: The number of months to add.
   :type months: int

   :param modifier: The rule to use to roll a resultant non-business day.
   :type modifier: Modifier

   :param roll: The day of the month to adjust to.
   :type roll: RollDay

   :param settlement: Enforce an associated settlement calendar, if *True* and if one exists.
   :type settlement: bool

   :rtype: datetime
