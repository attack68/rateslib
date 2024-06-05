.. add_bus_days
   =============

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.add_bus_days(date, days, settlement)

   Return a business date separated by `days` from an input business `date`.

   :param date: The original business date. Raise if a non-business date is given.
   :type date: datetime

   :param days: Number of business days to add.
   :type days: int

   :param settlement: If the calendar object contains an associated settlement calendar ensure the resultant date can settle transactions, if *True*.
   :type settlement: bool

   :rtype: datetime
