.. lag
   ==========

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.lag(date, days, settlement)

   Adjust a date by a number of business days, under lag rules.

   :param date: Input date to adjust.
   :type date: datetime

   :param days: Number of business days to add.
   :type days: int

   :param settlement: If *True*, also enforce adherence to an associated settlement calendar.
   :type settlement: bool

   :rtype: datetime

   .. rubric:: Notes

   **Lag rules** define the addition of business days to a date that is a non-business date:

   - Adding zero days will roll the date **forwards** to the next available business day.
   - Adding one day will roll the date **forwards** to the next available business day.
   - Subtracting one day will roll the date **backwards** to the previous available business day.

   Adding (or subtracting) further business days adopts the
   :meth:`~rateslib.calendars.Cal.add_bus_days` approach with a valid result.
