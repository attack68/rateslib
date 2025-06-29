.. _cal-doc:

.. ipython:: python
   :suppress:

   from rateslib.calendars import *
   from datetime import datetime as dt

************
Calendars
************

The ``rateslib.calendars`` module generates holiday calendars so that
business days are well defined, with additional objects for date manipulation.

Summary
*******

Classes
--------
.. autosummary::
   rateslib.calendars.Cal
   rateslib.calendars.NamedCal
   rateslib.calendars.UnionCal
   rateslib.calendars.Adjuster

Methods
-------
.. autosummary::
   rateslib.calendars.get_calendar
   rateslib.calendars.get_imm
   rateslib.calendars.next_imm
   rateslib.calendars.add_tenor
   rateslib.calendars.dcf


Why use 3 Calendar types?
**************************

Every calendar type has the same date manipulation methods, such as :meth:`~rateslib.calendars.Cal.is_bus_day` or
:meth:`~rateslib.calendars.Cal.bus_date_range`, for example. The different types provide
slightly different features.

**Cal**

The :class:`~rateslib.calendars.Cal` class is the *base* holiday calendar. It is a simple object for storing a list of
datetimes as the holidays and maintaining a record of what are the weekends (i.e. Saturdays and Sundays).
Its business day assessment is made by comparing against its list of ``holidays`` and ``week_mask``. *Cals*
provide the flexibility to create custom calendars.

.. ipython:: python

   cal = Cal(
       holidays=[dt(2023, 12, 25), dt(2023, 12, 26), dt(2024, 1, 1)],
       week_mask=[5, 6]
   )
   cal.bus_date_range(start=dt(2023, 12, 18), end=dt(2024, 1, 5))

For ease of use and performance *rateslib* has pre-defined and compiled a set of existing common fixed income
calendars for example the european *Target* calendar and the FED's settlement calendar.
See :ref:`default calendars <spec-defaults-calendars>`.

**UnionCal**

The :class:`~rateslib.calendars.UnionCal` class allows combinations of :class:`~rateslib.calendars.Cal`, to extend the
business day mathematics. This is required for multi-currency instruments. A *UnionCal* can contain, and replicate,
just one *Cal*.

.. ipython:: python

   union_cal = UnionCal(
       calendars=[cal],
       settlement_calendars=None
   )
   union_cal.bus_date_range(start=dt(2023, 12, 18), end=dt(2024, 1, 5))

**Calendar equivalence** checks that every business date and every potential settlement date are the same in both
calendars. For example, in this case we have that:

.. ipython:: python

   cal == union_cal

and these two calendar objects will perform exactly the same date manipulation functions.

**NamedCal**

The :class:`~rateslib.calendars.NamedCal` class is a wrapper for a *UnionCal*. It is a convenient object
because it will construct holiday calendars directly from *rateslib's* pre-defined list of *Cals* using a
**string parsing syntax**, which is suitable for multi-currency *Instruments*. This also
improves *serialization* as shown below.

*NamedCals* can only be used to load and combine the pre-compiled calendars. Custom calendars must be created
with the *Cal* and/or *UnionCal* objects.

Loading existing calendars
***************************

It is possible to load one of the :ref:`default calendars <spec-defaults-calendars>`
directly using a *NamedCal* as follows:

.. ipython:: python

   named_cal = NamedCal("tgt")

.. warning::

   Use defaults calendars at your own risk. Generally the repeated yearly holidays are
   accurate but a full list of ad-hoc and specialised holidays has may not necessarily be
   upto date.

Alternatively, the :meth:`~rateslib.calendars.get_calendar` method can be used (and is used internally)
to parse the different options a user might provide. This is more flexible because it
can return a calendar with no holidays on null input, or it can also load custom
calendars that have been dynamically added to *rateslib's* ``defaults.calendars`` object, or it
can also return *Cal* or *UnionCal* objects directly if preferred. This only has relevance for serialization.

.. ipython:: python

   cal = get_calendar("ldn", named=False)
   type(cal)
   named_cal = get_calendar("ldn", named=True)
   type(named_cal)
   cal == named_cal

Serialization
--------------

The `cal` and `named_cal` calendars created above are equivalent with reference to dates, even though they are
two different types. The difference is how these objects are serialized. `named_cal` will deserialize to the
*"ldn"* calendar that is defined as of the current version (and which may be updated from version to version),
whilst `cal` is statically defined for a list of dates (it will never change).

.. ipython:: python

   cal.to_json()
   named_cal.to_json()

These JSON strings will deserialize directly into the types from which they were constructed.

.. _settlement-cals:

Calendar combinations
**********************

Custom calendar combinations can be constructed with the :class:`~rateslib.calendars.UnionCal`
class. It requires a list of *Cal* objects to form the union of non-business dates,
and another, secondary list, of associated ``settlement_calendars``, to validate
calculated dates against.

Combined calendars can also be constructed automatically using **string parsing syntax**.
Comma separation forms a union of calendars. For example the appropriate calendar
for a EUR/USD cross-currency swap is *"tgt,nyc"* for TARGET and New York.

The appropriate holiday calendar to use for a EURUSD FX instrument, such as spot
determination is *"tgt|fed"*, which performs date manipulation under a TARGET calendar
but enforces associated settlement against the Fed settlement calendar. The associated settlement
calendar here is defined after the pipe operator.

.. ipython:: python

   # Combined calendar with no associated settlement calendar
   tgt_nyc = get_calendar("tgt,nyc")
   tgt_nyc.is_bus_day(dt(2009, 11, 11))
   tgt_nyc.is_settlement(dt(2009, 11, 11))

   # TARGET calendar enforcing New York settlement
   tgt_nyc_settle = get_calendar("tgt|nyc")
   tgt_nyc_settle.is_bus_day(dt(2009, 11, 11))
   tgt_nyc_settle.is_settlement(dt(2009, 11, 11))


Adding Custom Calendars to Defaults
**************************************

Custom calendars can be added to the ``defaults`` object and this allows the
:meth:`~rateslib.calendars.get_calendar` method to access it via string representation
in *Instrument* instantiation or or in other methods such as :meth:`~rateslib.calendars.add_tenor`.

Suppose we create a custom calendar which allows only Wednesdays to be business days.
We can then use this calendar to derive IMM dates in a month, although this is
obviously a pathological way of doing this, it is just shown for example purposes.

.. ipython:: python

   cal = Cal(holidays=[], week_mask=[0, 1, 3, 4, 5, 6])
   defaults.calendars["wednesdays"] = cal

   # The IMM date in March 2025 is..
   add_tenor(dt(2025, 3, 15), "0d", calendar="wednesdays", modifier="F")

Whenever the ``calendar`` argument is required for a method the string *'wednesdays'* could
now be freely used and would refer back to this object.

.. ipython:: python
   :suppress:

   defaults.reset_defaults()

Day count fractions (DCFs)
**************************

This module also contains a :meth:`~rateslib.calendars.dcf` method for calculating
day count fractions.
Review the API documentation for specific calculation details. Current DCF conventions
available are listed below:

.. ipython:: python

   from rateslib.calendars import _DCF
   print(_DCF.keys())
