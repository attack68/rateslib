RollDay
==========

.. currentmodule:: rateslib.scheduling

.. py:class:: RollDay

   Enumerable type for day roll types.

   **Values**

   - RollDay.Unspecified: the roll day will be inferred from in the input date's day.
   - RollDay.SoM: semantically equivalent to RollDay.Int(1).
   - RollDay.EoM: sematically equivalent to RollDay.Int(31).
   - RollDay.IMM: IMM dates in any given month.
   - RollDay.Int(_): A numeric day in the month, e.g. RollDay.Int(12).
