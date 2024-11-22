.. _cook-fixings-exposures-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from rateslib.calendars import get_imm
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Index
   from rateslib import defaults
   defaults.reset_defaults()

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

   irs = IRS(
       effective=dt(2000, 1, 10),
       termination="1w",
       spec="eur_irs",
       curves=estr,
       notional=2e6,
   )
   irs.fixings_table()

.. ipython:: python

   zcs = ZCS(
       effective=dt(2000, 1, 10),
       termination="4m",
       frequency="M",
       convention="act360",
       leg2_fixing_method="ibor",
       leg2_method_param=2,
       curves=[euribor1m, estr],
       notional=1.5e6,
   )
   zcs.fixings_table()

.. ipython:: python

   sbs = SBS(
       effective=dt(2000, 1, 6),
       termination="6m",
       spec="eur_sbs36",
       curves=[euribor3m, estr, euribor6m, estr],
       notional=3e6,
   )
   sbs.fixings_table()

.. ipython:: python

   iirs = IIRS(
       effective=dt(2000, 1, 10),
       termination="3m",
       frequency="M",
       index_base=100.0,
       index_lag=3,
       leg2_fixing_method="ibor",
       leg2_method_param=2,
       curves=[estr, estr, euribor1m, estr],
       notional=4e6,
   )
   iirs.fixings_table()

.. ipython:: python

   fra = FRA(
       effective=get_imm(code="H0"),
       termination=get_imm(code="M0"),
       roll="imm",
       spec="eur_fra3",
       curves=[euribor3m, estr],
       notional=5e6,
   )
   fra.fixings_table()

.. ipython:: python

   sofr = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.93}, calendar="nyc", id="sofr", convention="act360")
   xcs = XCS(
       effective=dt(2000, 1, 7),
       termination="9m",
       spec="eurusd_xcs",
       leg2_fixed=True,
       leg2_mtm=False,
       fixing_method="ibor",
       method_param=2,
       curves=[euribor3m, estr, sofr, sofr],
       notional=1e6,
   )
   xcs.fixings_table()

.. ipython:: python

   stir = STIRFuture(
       effective=get_imm(code="H0"),
       termination="3m",
       spec="eur_stir3",
       curves=[euribor3m],
       contracts=10,
   )
   stir.fixings_table()

.. ipython:: python

   frn = FloatRateNote(
      effective=dt(2000, 1, 13),
      termination="6m",
      frequency="Q",
      fixing_method="ibor",
      method_param=2,
      float_spread=120.0,
      curves=[euribor3m, estr]
   )
   frn.fixings_table()