.. _cook-bus252-conv:

.. ipython:: python
   :suppress:

   from rateslib import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Brazil's Bus252 Convention and Curve Calibration
*************************************************

The day count ``convention`` in Brazil is unconventional. Annual interest rates
are typically defined with **compounded rates** with day count fractions dependent
upon business days. This differs to the **simple period** rate definition often used
in G10 IRS. The following are equivalent (where *f* is usually set to 1 for annualized rates):

.. math::

   \underbrace{1 + d r}_{\text{simple rate}} = \underbrace{\left (1+\frac{r^{irr}}{f} \right)^{fd}}_{\text{compounded rate}}

Setup
------

When defining a :class:`~rateslib.curves.Curve` to calibrate, we specifically add the
``convention`` *'bus252'*, and will add a ``calendar`` specifically designed for Brazil
in 2025 and 2026.

.. ipython:: python

   holidays = [
       "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
       "2025-06-19", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20",
       "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
       "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15",
       "2026-11-20", "2026-12-25",
   ]
   bra = Cal(holidays=[dt.strptime(_, "%Y-%m-%d") for _ in holidays], week_mask=[5, 6])

   curve = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): 1.0,
           dt(2025, 11, 3): 1.0,
           dt(2026, 5, 1): 1.0,
       },
       convention="bus252",
       calendar=bra,
       interpolation="log_linear",
       id="curve",
   )

Instruments
------------

The *Instruments* used for calibration replicate the rates on DI1 futures. These are
zero coupon swaps, :class:`~rateslib.instruments.ZCS`, whose fixed rate definition is a
compounded rate.

The implied rates from the futures data are assumed to be 14%, 13.7% and 13.5%.

.. ipython:: python

   zcs_args = dict(frequency="A", calendar=bra, curves="curve", currency="brl", convention="bus252")
   solver = Solver(
       curves=[curve],
       instruments=[
           ZCS(dt(2025, 5, 15), dt(2025, 8, 1), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2025, 11, 3), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2026, 5, 1), **zcs_args),
       ],
       s=[14.0, 13.7, 13.5]
   )

Plotting
---------

.. ipython:: python

   curve.plot("1b")

.. plot::

   from rateslib import *
   import matplotlib.pyplot as plt

   holidays = [
       "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
       "2025-06-19", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20",
       "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
       "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15",
       "2026-11-20", "2026-12-25",
   ]
   bra = Cal(holidays=[dt.strptime(_, "%Y-%m-%d") for _ in holidays], week_mask=[5, 6])

   curve = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): 1.0,
           dt(2025, 11, 3): 1.0,
           dt(2026, 5, 1): 1.0,
       },
       convention="bus252",
       calendar=bra,
       interpolation="log_linear",
       id="curve",
   )

   zcs_args = dict(frequency="A", calendar=bra, curves="curve", currency="brl", convention="bus252")
   solver = Solver(
       curves=[curve],
       instruments=[
           ZCS(dt(2025, 5, 15), dt(2025, 8, 1), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2025, 11, 3), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2026, 5, 1), **zcs_args),
       ],
       s=[14.0, 13.7, 13.5]
   )

   fig, ax, line = curve.plot("1b")
   plt.show()
   plt.close()

This *Curve* demonstrate the traditional stepped interest rate structure. This is because the
*'log_linear'* ``interpolation`` has been applied on a **business day basis** in accordance with
the *'bus252'* ``convention`` and provided ``calendar``. But don't forget these are **simple**
rates. We adjust these in the final plot on this page.

As a demonstration of the difference in discount factors these can be plotted both for this
`curve` and a conventional *Curve* with the same node values. Under a business day, and
not calendar day, style the discount factors remain constant on a curve for a date which
is **not** a business day.

.. ipython:: python

   conventional = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): curve[dt(2025, 8, 1)],
           dt(2025, 11, 3): curve[dt(2025, 11, 3)],
           dt(2026, 5, 1): curve[dt(2026, 5, 1)],
       },
       convention="act365f",
       calendar=bra,
       interpolation="log_linear"
   )

   fig, ax = plt.subplots(1, 1)
   x, y1, y2 = [], [], []
   for date in bra.cal_date_range(dt(2025, 5, 15), dt(2026, 6, 15)):
       x.append(date)
       y1.append(curve[date])
       y2.append(conventional[date])

   ax.plot(x, y1)
   ax.plot(x, y2)

.. plot::

   from rateslib import *
   import matplotlib.pyplot as plt

   holidays = [
       "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
       "2025-06-19", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20",
       "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
       "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15",
       "2026-11-20", "2026-12-25",
   ]
   bra = Cal(holidays=[dt.strptime(_, "%Y-%m-%d") for _ in holidays], week_mask=[5, 6])

   curve = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): 1.0,
           dt(2025, 11, 3): 1.0,
           dt(2026, 5, 1): 1.0,
       },
       convention="bus252",
       calendar=bra,
       interpolation="log_linear",
       id="curve",
   )

   zcs_args = dict(frequency="A", calendar=bra, curves="curve", currency="brl", convention="bus252")
   solver = Solver(
       curves=[curve],
       instruments=[
           ZCS(dt(2025, 5, 15), dt(2025, 8, 1), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2025, 11, 3), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2026, 5, 1), **zcs_args),
       ],
       s=[14.0, 13.7, 13.5]
   )

   conventional = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): curve[dt(2025, 8, 1)],
           dt(2025, 11, 3): curve[dt(2025, 11, 3)],
           dt(2026, 5, 1): curve[dt(2026, 5, 1)],
       },
       convention="act365f",
       calendar=bra,
       interpolation="log_linear"
   )

   fig, ax = plt.subplots(1, 1)
   x, y1, y2 = [], [], []
   for date in bra.cal_date_range(dt(2025, 5, 15), dt(2025, 6, 15)):
       x.append(date)
       y1.append(curve[date])
       y2.append(conventional[date])

   ax.plot(x, y1)
   ax.plot(x, y2)
   plt.show()
   plt.close()

The rates displayed in plots depend upon their definition and day count fractions. The DFs on
both the `curve` and `conventional` *Curves* are **the same** but their O/N plots are
quite different because of this. We can even convert the O/N rates to compounded rates manually
as an additional comparison.

.. ipython:: python

   fig, ax, lines = curve.plot("1b", comparators=[conventional])
   y2 = ((1 + lines[0]._y / 25200)**252 - 1) * 100
   ax.plot( lines[0]._x, y2)
   ax.legend(["Bus252", "Act365", "Bus252 Compounded"])

.. plot::

   from rateslib import *
   import matplotlib.pyplot as plt

   holidays = [
       "2025-01-01", "2025-03-03", "2025-03-04", "2025-04-18", "2025-04-21", "2025-05-01",
       "2025-06-19", "2025-09-07", "2025-10-12", "2025-11-02", "2025-11-15", "2025-11-20",
       "2025-12-25", "2026-01-01", "2026-02-16", "2026-02-17", "2026-04-03", "2026-04-21",
       "2026-05-01", "2026-06-04", "2026-09-07", "2026-10-12", "2026-11-02", "2026-11-15",
       "2026-11-20", "2026-12-25",
   ]
   bra = Cal(holidays=[dt.strptime(_, "%Y-%m-%d") for _ in holidays], week_mask=[5, 6])

   curve = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): 1.0,
           dt(2025, 11, 3): 1.0,
           dt(2026, 5, 1): 1.0,
       },
       convention="bus252",
       calendar=bra,
       interpolation="log_linear",
       id="curve",
   )

   zcs_args = dict(frequency="A", calendar=bra, curves="curve", currency="brl", convention="bus252")
   solver = Solver(
       curves=[curve],
       instruments=[
           ZCS(dt(2025, 5, 15), dt(2025, 8, 1), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2025, 11, 3), **zcs_args),
           ZCS(dt(2025, 5, 15), dt(2026, 5, 1), **zcs_args),
       ],
       s=[14.0, 13.7, 13.5]
   )

   conventional = Curve(
       nodes={
           dt(2025, 5, 15): 1.0,
           dt(2025, 8, 1): curve[dt(2025, 8, 1)],
           dt(2025, 11, 3): curve[dt(2025, 11, 3)],
           dt(2026, 5, 1): curve[dt(2026, 5, 1)],
       },
       convention="act365f",
       calendar=bra,
       interpolation="log_linear"
   )

   fig, ax, lines = curve.plot("1b", comparators=[conventional])
   y2 = ((1 + lines[0]._y / 25200)**252 - 1)*100
   ax.plot( lines[0]._x, y2)
   ax.legend(["Bus252", "Act365", "Bus252 Compounded"])

   plt.show()
   plt.close()
