.. _cook-fixings-exposures-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Index

Fixings Exposures and Reset Ladders
*************************************

Every *Instruments* that has an exposure to a floating interest rate contains a method
that will obtain the risk and specific notional equivalent for an individual fixing.
This is a very useful display for fixed income rates derivative trading.

The column headers for the resultant *DataFrames* will be dynamically determined from the ``id``
of the *Curve* which forecasts the exposed fixing.

.. ipython:: python

   estr = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="estr", convention="act360")
   euribor1m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor1m", convention="act360")
   euribor3m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor3m", convention="act360")
   euribor6m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor6m", convention="act360")

.. ipython:: python

   irs = IRS(dt(2000, 1, 10), "1w", spec="eur_irs", curves=estr)
   irs.fixings_table()

.. ipython:: python

   sbs = SBS(dt(2000, 1, 6), "3m", frequency="Q", leg2_frequency="M", convention="act360", calendar="tgt", curves=[euribor3m, estr, euribor1m, estr])
   sbs.fixings_table()

.. ipython:: python

   sofr = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.93}, calendar="nyc", id="sofr", convention="act360")
   xcs = XCS(dt(2000, 1, 7), "9m", spec="eurusd_xcs", leg2_fixed=True, leg2_mtm=False, fixing_method="ibor", method_param=2, curves=[euribor3m, estr, sofr, sofr])
   xcs.fixings_table()