.. _schedule-doc:

.. ipython:: python
   :suppress:

   from rateslib.scheduling import *
   from datetime import datetime as dt

************
Schedule
************

The ``rateslib.scheduling`` module generates swap schedules.
Scheduling swaps is a surprisingly complex
issue, especially when one wants to infer some necessary parameters from the given
information. We will give examples of the basic :class:`~rateslib.scheduling.Schedule`
s and further explain some of the more complicated inference patterns.

Summary
*******

Classes
-------
.. autosummary::
   rateslib.scheduling.Schedule

Methods
-------
.. autosummary::
   rateslib.scheduling._check_regular_swap
   rateslib.scheduling._infer_stub_date

Scheduling Examples
********************

Regular Schedule
----------------------

A regular schedule under a generic business day calendar. Note how this
swap actually starts and end on the 1st of the month but the holiday adjusted
effective and termination dates are actually the 3rd and 2nd.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2022,1,1),
       termination=dt(2023,1,1),
       frequency="S",
       calendar="bus",
       payment_lag=1
   )
   schedule

If the same schedule is created with the adjusted dates input as the effective and
termination dates then a roll day is inferred, in this case as 2, creating a different
schedule to the above.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2022,1,3),
       termination=dt(2023,1,2),
       frequency="S",
       calendar="bus",
       payment_lag=1
   )
   schedule

The original schedule can be obtained by directly specifying the roll day and not
relying on roll day inference.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2022,1,3),
       termination=dt(2023,1,2),
       frequency="S",
       roll=1,
       calendar="bus",
       payment_lag=1
   )
   schedule

Defined Stubs
--------------

A schedule with specifically defined stubs.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2021,1,1),
       termination=dt(2021,10,1),
       frequency="Q",
       front_stub=dt(2021, 2, 26),
       back_stub=dt(2021, 8, 29),
       calendar="bus",
       payment_lag=1
   )
   schedule

Note that the above schedule must have a **regular swap** defined between stub dates.
In this case the roll, inferred as 29, allows this, and the unadjusted dates are then
adjusted under the business day holiday calendar to the provided stubs. Schedules that
cannot be inferred validly will raise.

Stub and roll generation can also be implied if the ``front_stub`` and/or ``back_stub``
are blank. Only one side can be inferred however so with a dual sided stub at least
one date must be given. In the following case *"FRONT"* suffices as the ``stub`` input
since the specific date is given, but *"LONGBACK"* provides the necessary detail for
inference. Without specifying roll here it would be inferred as 26, but an alternative,
valid value can be forced. Invalid combinations (those that do not permit regular swaps
between stub dates) raise errors.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2021, 1, 1),
       termination=dt(2021, 10, 1),
       frequency="Q",
       front_stub=dt(2021, 2, 26),
       stub="FRONTLONGBACK",
       roll=30,
       calendar="bus",
       payment_lag=1
   )
   schedule

.. ipython:: python

   try:
       Schedule(
           effective=dt(2021, 1, 1),
           termination=dt(2021, 10, 1),
           frequency="Q",
           front_stub=dt(2021, 2, 26),
           stub="FRONTLONGBACK",
           roll=25,
           calendar="bus",
           payment_lag=1
       )
   except ValueError as e:
       print(e)

Simple Inference
-----------------

One-sided stub inference can also be made if no stub dates are defined.

.. ipython:: python

   schedule = Schedule(
       effective=dt(2021, 1, 1),
       termination=dt(2021, 7, 15),
       frequency="Q",
       stub="SHORTFRONT",
       calendar="bus",
       payment_lag=1
   )
   schedule

.. ipython:: python

   schedule = Schedule(
       effective=dt(2021, 1, 1),
       termination=dt(2021, 7, 15),
       frequency="Q",
       stub="LONGBACK",
       calendar="bus",
       payment_lag=1
   )
   schedule
