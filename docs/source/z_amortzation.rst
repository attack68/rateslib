.. _cook-amortization-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Applying Amortization to Instruments
******************************************************

The :class:`~rateslib.legs.FixedLeg` and :class:`~rateslib.legs.FloatLeg` classes both
have ``amortization`` as an input argument which defines the reduction (or addition) of
a notional amount each period.

.. ipython:: python

   curve = Curve({dt(2000, 1, 1): 1.0, dt(2010, 1, 1): 0.65})

.. ipython:: python

   fxl = FixedLeg(
       effective=dt(2000, 1, 1),
       termination="1y",
       frequency="Q",
       notional=10e6
       amortization=1e6
   )
   fxl.cashflows(curve)

.. ipython:: python

   irs = IRS(
       effective=dt(2022, 1, 15),
       termination=dt(2022, 11, 15),
       front_stub=dt(2022, 3, 15),
       back_stub=dt(2022, 9, 15),
       frequency="Q",
       fixed_rate=1.5,
       currency="eur",
       leg2_fixing_method="ibor"
   )

