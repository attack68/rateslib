.. _cook-ir-vol-time-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from rateslib.solver import Solver
   from rateslib import calendars
   from itertools import product
   from rateslib.volatility import IRSabrCube, IRSplineCube, IRSplineSmile, IRSabrSmile
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Series
   import pandas as pd

IR Volatility Time To Expiry Remapping
**********************************************************************

This page presents examples for working with time to expiry for IR volatility products.

  **Key Points**

  - Every *time to expiry* is an Act365 calendar day measure unless remapped.
  - At each ``expiry`` on any *Cube* *time to expiry* is Act365 calendar day measure assuming the associated
    volatility is calibrated to market.
  - Any intermediate *time to expiry* between the chosen ``expiries`` on a *Cube* can be remapped.

Introduction
-------------

Every *IR volatility* pricing object has an ``eval_date`` as part of its ``meta`` parameters.
This allows any :class:`~rateslib.volatility._BaseIRSmile` to make a natural measure of time to expiry
using the equation:

.. math::

   t = \frac{days(expiry - eval date)}{365}

When a :class:`~rateslib.volatility._BaseIRSmile` yields a volatility value for a specific strike,
that volatility value is assumed to be associated with that *time to expiry* that
that :class:`~rateslib.volatility._BaseIRSmile` calculates.

Most of the time a user will not need to be aware of that. Lets create a basic swaption and analyse
different pricing models. For example:

.. ipython:: python

   curve = Curve({dt(2001, 1, 1): 1.0, dt(2004, 1, 1): 0.90}, convention="act360", calendar="nyc")
   iro = IRSCall(expiry=dt(2002, 1, 1), tenor="1y", irs_series="usd_irs", strike=3.30)

.. tabs::

   .. group-tab:: IRSplineSmile

      .. ipython:: python

         irss = IRSplineSmile(
             eval_date=dt(2001, 1, 1),
             expiry=dt(2002, 1, 1),
             tenor="1y",
             nodes={-25.0: 52, 0: 50, 25: 53},
             k=4,
             irs_series="usd_irs"
         )
         print(irss.get_from_strike(k=2.4, f=2.1))
         print(iro.rate(curves=curve, vol=irss, metric="percentnotional"))

   .. group-tab:: IRSabrSmile

      .. ipython:: python

         irss = IRSabrSmile(
             eval_date=dt(2001, 1, 1),
             expiry=dt(2002, 1, 1),
             tenor="1y",
             nodes={"alpha": 0.35, "rho": -0.05, "nu": 0.65},
             beta=0.5,
             irs_series="usd_irs",
         )
         print(irss.get_from_strike(k=2.4, f=2.1))
         print(iro.rate(curves=curve, vol=irss, metric="percentnotional"))

The pair of values here :math:`(t, \sigma)` are used in pricing models such as the Black76 or Bachelier model directly.

Time Scaling
--------------

It is possible, however, to apply a scaling parameter to the calendar day measure to arrive at a different
*time to expiry*. Doing so yields a pair :math:`(\hat{t}, \hat{\sigma})`

.. math::

   \hat{t} = \xi t

.. tabs::

   .. group-tab:: IRSplineSmile

      .. ipython:: python

         irss = IRSplineSmile(
             eval_date=dt(2001, 1, 1),
             expiry=dt(2002, 1, 1),
             tenor="1y",
             nodes={-25.0: 52, 0: 50, 25: 53},
             k=4,
             irs_series="usd_irs",
             time_scalar=0.98,
         )
         print(irss.get_from_strike(k=2.4, f=2.1))
         print(iro.rate(curves=curve, vol=irss, metric="percentnotional"))

   .. group-tab:: IRSabrSmile

      .. ipython:: python

         irss = IRSabrSmile(
             eval_date=dt(2001, 1, 1),
             expiry=dt(2002, 1, 1),
             tenor="1y",
             nodes={"alpha": 0.35, "rho": -0.05, "nu": 0.65},
             beta=0.5,
             irs_series="usd_irs",
             time_scalar=0.98,
         )
         print(irss.get_from_strike(k=2.4, f=2.1))
         print(iro.rate(curves=curve, vol=irss, metric="percentnotional"))

Working with a Cube
----------------------

Typically the *time scalar* is not a quantity one will add to a *Smile* directly.
Instead it exists to allow *Cubes* to handle time interpolation.
The ``weights`` argument on a *Cube* can apportion volatility to specific dates in between
chosen ``expiries``. It is **assumed** that on every given expiry the time scalar equals one
and the *Cube* is calibrated to market *Instruments*.

.. tabs::

   .. tab:: Calendar Days

      .. ipython:: python

         irsc1 = IRSabrCube(
             eval_date=dt(2001, 1, 1),
             expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
             tenors=["1y"],
             alpha=0.35,
             beta=0.5,
             rho=-0.05,
             nu=0.45,
             irs_series="usd_irs",
         )

   .. tab:: Business Days

      .. ipython:: python

         nyc = calendars.get("nyc")
         weights = Series(  # set the weight of non-business days to zero
             index=[_ for _ in nyc.cal_date_range(dt(2001, 1, 1), dt(2001, 8, 1)) if nyc.is_non_bus_day(_)],
             data=0.0
         )
         irsc2 = IRSabrCube(
             eval_date=dt(2001, 1, 1),
             expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
             tenors=["1y"],
             alpha=0.35,
             beta=0.5,
             rho=-0.05,
             nu=0.45,
             irs_series="usd_irs",
             weights=weights,
         )

   .. tab:: Semi-Business Days

      .. ipython:: python

         weights2 = Series(  # set the weight of non-business days to 0.5
             index=[_ for _ in nyc.cal_date_range(dt(2001, 1, 1), dt(2001, 8, 1)) if nyc.is_non_bus_day(_)],
             data=0.5
         )
         irsc3 = IRSabrCube(
             eval_date=dt(2001, 1, 1),
             expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
             tenors=["1y"],
             alpha=0.35,
             beta=0.5,
             rho=-0.05,
             nu=0.45,
             irs_series="usd_irs",
             weights=weights2,
         )

Prices of Options
-------------------

With the different models above we plot the prices of ATM Payer Swaptions. In fact these graphs show the
differences in prices of percent of notional for an option of every expiry date. After the end of the ``weights``
*Series* the prices converge as both models fall back to calendar day type.

.. ipython:: python

   x, y, y2 = [], [], []
   for expiry in nyc.cal_date_range(dt(2001, 1, 5), dt(2001, 9, 1)):
       iro = IRSCall(
           expiry=expiry,
           tenor="1y",
           strike="atm",
           irs_series="usd_irs",
       )
       x.append(expiry)
       y.append(iro.rate(curves=curve, vol=irsc1, metric="percentnotional") - iro.rate(curves=curve, vol=irsc2, metric="percentnotional"))
       y2.append(iro.rate(curves=curve, vol=irsc1, metric="percentnotional") - iro.rate(curves=curve, vol=irsc3, metric="percentnotional"))


.. plot::

   from rateslib import dt, Curve, IRSabrCube, calendars, IRSCall
   from pandas import Series

   curve = Curve({dt(2001, 1, 1): 1.0, dt(2004, 1, 1): 0.90}, convention="act360", calendar="nyc")
   irsc1 = IRSabrCube(
       eval_date=dt(2001, 1, 1),
       expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
       tenors=["1y"],
       alpha=0.35,
       beta=0.5,
       rho=-0.05,
       nu=0.45,
       irs_series="usd_irs",
   )
   nyc = calendars.get("nyc")
   weights = Series(  # set the weight of non-business days to zero
       index=[_ for _ in nyc.cal_date_range(dt(2001, 1, 1), dt(2001, 8, 1)) if nyc.is_non_bus_day(_)],
       data=0.0
   )
   weights2 = Series(  # set the weight of non-business days to 0.5
       index=[_ for _ in nyc.cal_date_range(dt(2001, 1, 1), dt(2001, 8, 1)) if nyc.is_non_bus_day(_)],
       data=0.5
   )
   irsc2 = IRSabrCube(
       eval_date=dt(2001, 1, 1),
       expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
       tenors=["1y"],
       alpha=0.35,
       beta=0.5,
       rho=-0.05,
       nu=0.45,
       irs_series="usd_irs",
       weights=weights,
   )
   irsc3 = IRSabrCube(
       eval_date=dt(2001, 1, 1),
       expiries=[dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)],
       tenors=["1y"],
       alpha=0.35,
       beta=0.5,
       rho=-0.05,
       nu=0.45,
       irs_series="usd_irs",
       weights=weights2,
   )
   x, y, y2 = [], [], []
   for expiry in nyc.cal_date_range(dt(2001, 1, 5), dt(2001, 9, 1)):
       iro = IRSCall(
           expiry=expiry,
           tenor="1y",
           strike="atm",
           irs_series="usd_irs",
       )
       x.append(expiry)
       y.append(iro.rate(curves=curve, vol=irsc1, metric="percentnotional") - iro.rate(curves=curve, vol=irsc2, metric="percentnotional"))
       y2.append(iro.rate(curves=curve, vol=irsc1, metric="percentnotional") - iro.rate(curves=curve, vol=irsc3, metric="percentnotional"))

   from matplotlib import pyplot as plt
   fig, ax = plt.subplots(1,1)
   ax.plot(x,y)
   ax.plot(x,y2)
   ax.scatter([dt(2001, 2, 1), dt(2001, 3, 1), dt(2001, 4, 1), dt(2001, 7, 1)], [0, 0, 0, 0], s=25, c='r')
   plt.show()
   plt.close()