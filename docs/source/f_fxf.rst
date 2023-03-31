.. _fxf-doc:

.. ipython:: python
   :suppress:

   from rateslib.fx import *
   from datetime import datetime as dt

*****************
FX Forward Rates
*****************

Basic spot :class:`~rateslib.fx.FXRates` are extended using discount factor based
:class:`~rateslib.curves.Curve` s to derive
arbitrage free forward FX rates. The basic :class:`~rateslib.fx.FXForwards` class is
summarised below,

.. autosummary::
   rateslib.fx.FXForwards
   rateslib.fx.FXForwards.rate
   rateslib.fx.FXForwards.swap
   rateslib.fx.FXForwards.curve
   rateslib.fx.FXForwards.plot

Introduction
------------

When calculating forward FX rates the following information is required;

  - The **spot** or **immediate FX rate** observable in the interbank market.
  - The **interest rates** in each currency to derive an interest rate parity
    expression.
  - The supply and demand factor, that impacts market **FX swap** or **cross-currency
    swap** price dynamics.

Thus the :class:`~rateslib.fx.FXForwards` class requires this information for instantiation.
If we suppose that the third element, the supply and demand factor from tenor
cross-currency markets is missing (or is zero), then we can instantiate the
class with just information from the first two elements.

.. ipython:: python

   # This is the spot FX exchange rates
   fx_rates = FXRates({"eurusd": 1.05}, settlement=dt(2022, 1, 3), base="usd")
   # These are the interest rate curves in EUR and USD
   usd_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965})
   eur_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985})
   fx_curves = {
       "usdusd": usd_curve,
       "eureur": eur_curve,
       "eurusd": eur_curve,  #  <- This is the same as "eureur" since no third factor
   }
   fxf = FXForwards(fx_rates, fx_curves)

With the class instantiated we can use it to calculate forward FX rates.

.. ipython:: python

   fxf.rate("eurusd", dt(2022, 9,15))

To explicitly verify this we can make this calculation manually, by extracting
relevant data from the curves.

.. ipython:: python

   usd_curve.rate(dt(2022, 1, 3), dt(2022, 9, 15))
   eur_curve.rate(dt(2022, 1, 3), dt(2022, 9, 15))
   dcf(dt(2022, 1, 3), dt(2022, 9, 15), "act360")

.. math::

   f_{EURUSD, i} = \frac{1 + d_i r_{USD, i}}{1 + d^*_i r^*_{EUR, i}} f_{EURUSD, i-1} = \frac{1 + 0.708 \times 0.03558}{1+0.708 \times 0.01499} \times 1.05 = 1.06515


Cross-Currency Swap and FX Swap Basis
--------------------------------------

In this example we will expand the above by adding the third component.
Suppose that rates in USD are 3.5%, in EUR 1.5% and in GBP 2%. The 1Y cross-currency basis
swap rates are EUR/USD (ESTR/SOFR) -20bps and GBP/USD (SONIA/SOFR) -30bps.
The following gives an approximate representation of this market.

.. ipython:: python
   :okwarning:

   fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.20}, settlement = dt(2022, 1, 3))
   fxf = FXForwards(fxr, {
       "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
       "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
       "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
       "gbpgbp": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.970}),
       "gbpusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.973})
   })

If we compare this to the above section the forward FX rates for EURUSD is slightly
different now that the third component is accounted for with an amended `"eurusd"`
discount curve.

.. ipython:: python

   fxf.rate("eurusd", dt(2022, 9, 15))

We can repeat the above manual calculation with the necessary adjustments.

.. ipython:: python

   fxf.fx_curves["eurusd"].rate(dt(2022, 1, 3), dt(2022, 9, 15))

.. math::

   f_{EURUSD, i} = \frac{1 + d_i r_{USDUSD, i}}{1 + d^*_i r^*_{EURUSD, i}} f_{EURUSD, i-1} = \frac{1 + 0.708 \times 0.03558}{1+0.708 \times 0.01297} \times 1.05 = 1.06666

Visualization
--------------

The :meth:`~rateslib.fx.FXForwards.plot` method exists for the
:class:`~rateslib.fx.FXForwards` class. We can plot the EURUSD
forward FX rates. Since our curves only contain one flat rate the FX forward rate
reflects a straight upward line when plotted for all settlement dates in the window.

.. ipython:: python
   :okwarning:

   fxf.plot("eurusd")

.. plot::

   from rateslib.curves import *
   from rateslib.fx import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.20}, settlement = dt(2022, 1, 3))
   fxf = FXForwards(fxr, {
       "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
       "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
       "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
       "gbpgbp": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.970}),
       "gbpusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.973})
   })
   fig, ax, line = fxf.plot("eurusd")
   plt.show()

ProxyCurves and Discounting
----------------------------

In a multi-currency framework there are often many *intrinsic* discount curves that can
be constructed that are not necessary for the initial construction of the
:class:`~rateslib.fx.FXForwards` class. For example the discount curve for GBP cashflows
discounted under a EUR collateral CSA (credit support annex), the "gbpeur" curve is
not provided at initialisation, nor is the "eurgbp" curve.

In these circumstances the :meth:`rateslib.fx.FXForwards.curve` will derive the
combination of existing curves that can be combined to yield required DFs on-the-fly.
This creates a ``ProxyCurve``.

In the above framework GBP is the cheapest to deliver collateral, and USD is the
most expensive. We can observe this
by calculating the curves in any cash currency for all collateral currencies
and plotting. This is demonstrated below.

.. ipython:: python
   :okwarning:

   type(fxf.curve("eur", "eur"))
   type(fxf.curve("eur", "usd"))
   type(fxf.curve("eur", "gbp"))
   fxf.curve("eur", "eur").plot("1d", labels=["eur", "usd", "gbp"], comparators=[
       fxf.curve("eur", "usd"),
       fxf.curve("eur", "gbp")
   ])

.. plot::

   from rateslib.curves import *
   from rateslib.fx import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   fxr = FXRates({"eurusd": 1.05, "gbpusd": 1.20}, settlement = dt(2022, 1, 3))
   fxf = FXForwards(fxr, {
       "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
       "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
       "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
       "gbpgbp": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.970}),
       "gbpusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.973})
   })
   fig, ax, line = fxf.curve("eur", "eur").plot("1d", comparators=[fxf.curve("eur", "usd"), fxf.curve("eur", "gbp")], labels=["eur", "usd", "gbp"])
   plt.show()
