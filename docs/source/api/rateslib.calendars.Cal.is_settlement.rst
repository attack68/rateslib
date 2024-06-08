.. is_settlement
   ================

.. .. currentmodule:: rateslib.calendars

.. py:method:: Cal.is_settlement(date)

   Return whether the `date` is a business day in an associated settlement.

   .. note::

      *Cal* objects will always return *True*, since they do not contain any
      associated settlement calendars. This method is provided only for API consistency.

   :param date: Date to test.
   :type date: datetime

   :rtype: bool
