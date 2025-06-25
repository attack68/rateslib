.. _cook-swpm-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Building a Conventional Par Tenor Based SOFR Curve
******************************************************

Many platforms and providers offer straight forward ways of constructing IR curves from market
data. Without introducing features such as meeting dates or turns we can replicate some of
those simpler models here. See the image below which contains the market data we will use.
This consists of purely par tenor IRSs.

.. image:: _static/sofr_swpm_1.PNG
  :alt: SOFR Curve

We can replicate the input data for the :class:`~rateslib.curves.Curve` in a table as follows:

.. ipython:: python

   data = DataFrame({
       "Term": ["1W", "2W", "3W", "1M", "2M", "3M", "4M", "5M", "6M", "7M", "8M", "9M", "10M", "11M", "12M", "18M", "2Y", "3Y", "4Y", "5Y", "6Y", "7Y", "8Y", "9Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y", "40Y"],
       "Rate": [5.309,5.312,5.314,5.318,5.351,5.382,5.410,5.435,5.452,5.467,5.471,5.470,5.467,5.457,5.445,5.208,4.990,4.650,4.458,4.352,4.291,4.250,4.224,4.210,4.201,4.198,4.199,4.153,4.047,3.941,3.719],
   })
   data["Termination"] = [add_tenor(dt(2023, 9, 29), _, "F", "nyc") for _ in data["Term"]]
   with option_context("display.float_format", lambda x: '%.6f' % x):
       print(data)

We will configure DF ``nodes`` dates to be on the termination date of the swaps:

.. ipython:: python

   sofr = Curve(
       id="sofr",
       convention="Act360",
       calendar="nyc",
       modifier="MF",
       interpolation="log_linear",
       nodes={
           **{dt(2023, 9, 27): 1.0},  # <- this is today's DF,
           **{_: 1.0 for _ in data["Termination"]},
       }
   )

Now we will calibrate the curve to the given swap market prices, using a global
:class:`~rateslib.solver.Solver`, passing in the calibrating instruments and rates.

.. ipython:: python

   solver = Solver(
       curves=[sofr],
       instruments=[IRS(dt(2023, 9, 29), _, spec="usd_irs", curves="sofr") for _ in data["Termination"]],
       s=data["Rate"],
       instrument_labels=data["Term"],
       id="us_rates",
   )
   data["DF"] = [float(sofr[_]) for _ in data["Termination"]]
   with option_context("display.float_format", lambda x: '%.6f' % x):
       print(data)

Next we will create an IRS and generate the metrics for *npv*, *delta* (DV01), *gamma* and
*analytic delta* (PV01).

.. ipython:: python

   irs = IRS(
       effective=dt(2023, 11, 21),
       termination=dt(2025, 2, 21),
       notional=-100e6,
       fixed_rate=5.40,
       curves="sofr",
       spec="usd_irs",
   )
   irs.npv(solver=solver)
   irs.delta(solver=solver).sum()
   irs.gamma(solver=solver).sum().sum()
   irs.analytic_delta(curve=sofr)

Finally we can double check the *cashflows* and *cashflows_table* of the swap.

.. ipython:: python

   irs.cashflows_table(solver=solver)
   irs.cashflows(solver=solver)
