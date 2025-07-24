.. _schedule-doc:

.. ipython:: python
   :suppress:

   from rateslib.scheduling import *
   from datetime import datetime as dt

************
Schedule
************

The ``rateslib.scheduling`` module generates common financial instrument schedules.
Scheduling is a surprisingly complex
issue, especially when one wants to infer some necessary parameters from the given
information.

The :class:`~rateslib.scheduling.Schedule` object has an **original** set of available input types,
used since the initial version of *rateslib* and a **core** set of available input types which
more closely align with the Rust re-implementation after version 2.0. These can be intermixed,
but for demonstration purposes this page uses core inputs.

.. tabs::

   .. tab:: Original Inputs

      The **original inputs** allow for a more UI friendly input for the most common schedules.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 15),
             dt(2001, 1, 1),
             "Q",
             stub="ShortFront",
             modifier="MF",
             payment_lag=2,
             calendar="tgt",
         )
         print(s)

   .. tab:: Core Inputs

      The **core inputs** utilise the Rust objects directly and may provide more flexibility.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 15),
             dt(2001, 1, 1),
             Frequency.Months(3, RollDay.Day(1)),
             stub=StubInference.ShortFront,
             modifier=Adjuster.ModifiedFollowing(),
             payment_lag=Adjuster.BusDaysLagSettle(2),
             calendar=NamedCal("tgt"),
         )
         print(s)


Summary
*******

Classes
-------
.. autosummary::
   rateslib.scheduling.Schedule
   rateslib.scheduling.Frequency
   rateslib.scheduling.RollDay
   rateslib.scheduling.Adjuster
   rateslib.scheduling.StubInference

Scheduling Examples
********************

The following scheduling patterns are possible to construct in *rateslib*.

- A **regular schedule**, which does not contain any stub periods.
- **Irregular schedules**, which consist of the following:

  - A **single stub** period, be it a short or long stub.
  - **Two stub** periods, combining any short and long varieties.
  - A **front stub** and a **regular schedule**.
  - A **regular schedule** and a **back stub**.
  - A **front stub** and a **regular schedule** and a **back stub**.


The below tabs give an example of each construction type. To minimise the complexity of these
examples all dates are given in their *unadjusted* form, which is how they should be given to
avoid any inference.

.. tabs::

   .. tab:: Regular

      The unadjusted ``effective`` and ``termination`` dates perfectly divide the ``frequency``.
      In this instance any ``stub`` inference parameter is unused.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2001, 1, 1),
             Frequency.Months(3, RollDay.Day(1)),
             stub=StubInference.ShortFront,
         )
         print(s)

   .. tab:: One Short Stub

      The ``stub`` inference parameter is explicitly set to *None* here.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 2, 15),
             Frequency.Months(3, RollDay.Day(1)),
             stub=None,
         )
         print(s)

   .. tab:: One Long Stub

      The ``stub`` inference parameter is explicitly set to *None* here.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 5, 15),
             Frequency.Months(3, RollDay.Day(1)),
             stub=None,
         )
         print(s)

   .. tab:: Two Stubs

      Both the ``front_stub`` and ``back_stub`` dates must be equivalent. ``stub`` inference is set
      to *None*.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 5, 15),
             Frequency.Months(3, None),
             front_stub=dt(2000, 1, 20),
             back_stub=dt(2000, 1, 20),
             stub=None,
         )
         print(s)

   .. tab:: Front Stub

      Set a ``front_stub`` and ensure the subsequent ``termination`` aligns with a regular schedule.
      Or permit ``stub`` inference.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 11, 15),
             Frequency.Months(3, None),
             stub=StubInference.ShortFront,
         )
         print(s)

   .. tab:: Back Stub

      Set a ``back_stub`` and ensure the preliminary ``effective`` aligns with a regular schedule.
      Or permit ``stub`` inference.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 11, 15),
             Frequency.Months(3, None),
             stub=StubInference.LongBack,
         )
         print(s)

   .. tab:: All

      Stub inference can only be applied to one side. Either supply both ``front_stub`` and
      ``back_stub`` or supply one and permit ``stub`` inference on the remaining side.

      .. ipython:: python

         s = Schedule(
             dt(2000, 1, 1),
             dt(2000, 11, 15),
             Frequency.Months(3, None),
             front_stub=dt(2000, 1, 21),
             stub=StubInference.ShortBack,
         )
         print(s)

Construction elements
***********************

A :class:`~rateslib.scheduling.Schedule` in *rateslib* is characterised by three major attributes:

- its **uschedule** which is a list of *unadjusted* dates defining its unambiguous skeletal
  structure.
- its **aschedule** which applies the ``modifier`` as an accrual :class:`~rateslib.scheduling.Adjuster`
  to adjust the *unadjusted* dates into *adjusted accrual dates* for defining its accrual periods.
- its **pschedule** which applies the ``payment_lag`` as a secondary :class:`~rateslib.scheduling.Adjuster`
  to adjust each *accrual date* to determine a physical payment, or cashflow settlement, date.Adjuster

All of the input arguments to a :class:`~rateslib.scheduling.Schedule` fit into the logic for
yielding these three components.
