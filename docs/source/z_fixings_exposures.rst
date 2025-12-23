.. _cook-fixings-exposures-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   from rateslib.scheduling import get_imm
   from rateslib.instruments import STIRFuture
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context, Index
   from rateslib import defaults
   defaults.reset_defaults()

Fixings Exposures and Reset Ladders
*************************************

Every *Instrument* that has an exposure to a floating interest rate contains a method
that will obtain the risk and specific notional equivalent for an individual fixing.
This is a very useful display for fixed income rates derivative trading.

The column headers for the resultant *DataFrames* will be dynamically determined from the ``id``
of the *Curve* which forecasts the exposed fixing.

.. ipython:: python

   # Setup some Curves against which fixing exposure will be measured

   estr = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="estr", convention="act360")
   euribor1m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor1m", convention="act360")
   euribor3m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor3m", convention="act360")
   euribor6m = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.95}, calendar="tgt", id="euribor6m", convention="act360")
   eur_cpi = Curve({dt(1999, 10, 1): 1.0, dt(2005, 1, 1): 0.92}, convention="act360", id="eur_cpi", index_base=100.0, index_lag=0)

.. tabs::

   .. tab:: IRS

      An *IRS* has exposure only via its :class:`~rateslib.legs.FloatLeg`.

      .. ipython:: python

         irs = IRS(
             effective=dt(2000, 1, 10),
             termination="1w",
             spec="eur_irs",
             curves=estr,
             notional=2e6,
         )
         irs.local_analytic_rate_fixings()

   .. tab:: ZCS

      A *ZCS* has exposure only via its :class:`~rateslib.legs.ZeroFloatLeg`.

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
         zcs.local_analytic_rate_fixings()

   .. tab:: SBS

      An *SBS* has exposure on each one of its two :class:`~rateslib.legs.FloatLeg`.

      .. ipython:: python

         sbs = SBS(
             effective=dt(2000, 1, 6),
             termination="6m",
             spec="eur_sbs36",
             curves=[euribor3m, estr, euribor6m, estr],
             notional=3e6,
         )
         sbs.local_analytic_rate_fixings()

   .. tab:: IIRS

      An *IIRS* has exposure only via its :class:`~rateslib.legs.FloatLeg`.

      .. ipython:: python

         iirs = IIRS(
             effective=dt(2000, 1, 10),
             termination="3m",
             frequency="M",
             index_base=100.0,
             index_lag=3,
             leg2_fixing_method="ibor",
             leg2_method_param=2,
             curves=[eur_cpi, estr, euribor1m, estr],
             notional=4e6,
         )
         iirs.local_analytic_rate_fixings()

   .. tab:: FRA

      A *FRA* has exposure only via its modified :class:`~rateslib.periods.FloatPeriod`.

      .. ipython:: python

         fra = FRA(
             effective=get_imm(code="H0"),
             termination=get_imm(code="M0"),
             roll="imm",
             spec="eur_fra3",
             curves=[euribor3m, estr],
             notional=5e6,
         )
         fra.local_analytic_rate_fixings()

   .. tab:: XCS

      A *XCS* has exposure only via its :class:`~rateslib.legs.FloatLeg` or
      :class:`~rateslib.legs.FloatLegMtm`. Any *FixedLegs* will not contribute.

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
         xcs.local_analytic_rate_fixings()

   .. tab:: STIRFuture

      A *STIRFuture* has exposure only due to its modified :class:`~rateslib.periods.FloatPeriod`.

      .. ipython:: python

         stir = STIRFuture(
             effective=get_imm(code="H0"),
             termination="3m",
             spec="eur_stir3",
             curves=[euribor3m],
             contracts=10,
         )
         stir.local_analytic_rate_fixings()

   .. tab:: FRN

      A *FloatRateNote* has exposure only to its :class:`~rateslib.legs.FloatLeg`.

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
         frn.local_analytic_rate_fixings()

..
    Performance: ``approximate`` and ``right``
    --------------------------------------------

    Calculating fixings exposures, particularly for every daily RFR rate, is an expensive calculation.
    The performance can be improved by firstly using ``approximate`` which yields almost exactly
    the same results but performs a faster, more generic calculation, and also using the ``right``
    bound, which avoids doing any calculation for exposures out-of-scope.

    .. tabs::

       .. tab:: IRS

          .. ipython:: python

             irs = IRS(
                 effective=dt(2000, 1, 10),
                 termination="4y",
                 spec="eur_irs",
                 curves=estr,
                 notional=2e6,
             )
             irs.local_analytic_rate_fixings(approximate=True, right=dt(2000, 1, 24))

       .. tab:: ZCS

          .. ipython:: python

             zcs = ZCS(
                 effective=dt(2000, 1, 10),
                 termination="1y",
                 frequency="M",
                 convention="act360",
                 leg2_fixing_method="ibor",
                 leg2_method_param=2,
                 curves=[euribor1m, estr],
                 notional=1.5e6,
             )
             zcs.fixings_table(approximate=True, right=dt(2000, 3, 17))

       .. tab:: SBS

          .. ipython:: python

             sbs = SBS(
                 effective=dt(2000, 1, 6),
                 termination="1y",
                 spec="eur_sbs36",
                 curves=[euribor3m, estr, euribor6m, estr],
                 notional=3e6,
             )
             sbs.fixings_table(approximate=True, right=dt(2000, 8, 16))

       .. tab:: IIRS

          .. ipython:: python

             iirs = IIRS(
                 effective=dt(2000, 1, 10),
                 termination="1y",
                 frequency="M",
                 index_base=100.0,
                 index_lag=3,
                 leg2_fixing_method="ibor",
                 leg2_method_param=2,
                 curves=[estr, estr, euribor1m, estr],
                 notional=4e6,
             )
             iirs.fixings_table(approximate=True, right=dt(2000, 8, 16))

       .. tab:: FRA

          .. ipython:: python

             fra = FRA(
                 effective=get_imm(code="H0"),
                 termination=get_imm(code="M0"),
                 roll="imm",
                 spec="eur_fra3",
                 curves=[euribor3m, estr],
                 notional=5e6,
             )
             fra.fixings_table(approximate=True, right=dt(2000, 8, 16))

       .. tab:: XCS

          .. ipython:: python

             sofr = Curve({dt(2000, 1, 1): 1.0, dt(2005, 1, 1): 0.93}, calendar="nyc", id="sofr", convention="act360")
             xcs = XCS(
                 effective=dt(2000, 1, 7),
                 termination="2y",
                 spec="eurusd_xcs",
                 leg2_fixed=True,
                 leg2_mtm=False,
                 fixing_method="ibor",
                 method_param=2,
                 curves=[euribor3m, estr, sofr, sofr],
                 notional=1e6,
             )
             xcs.fixings_table(approximate=True, right=dt(2000, 8, 16))

       .. tab:: STIRFuture

          .. ipython:: python

             stir = STIRFuture(
                 effective=get_imm(code="H0"),
                 termination="3m",
                 spec="eur_stir3",
                 curves=[euribor3m],
                 contracts=10,
             )
             stir.local_analytic_rate_fixings(curves=estr)

       .. tab:: FRN

          .. ipython:: python

             frn = FloatRateNote(
                 effective=dt(2000, 1, 13),
                 termination="2y",
                 frequency="Q",
                 fixing_method="ibor",
                 method_param=2,
                 float_spread=120.0,
                 curves=[euribor3m, estr]
             )
             frn.fixings_table(approximate=True, right=dt(2000, 8, 16))

..
    Aggregation of fixings exposures
    ---------------------------------

    Adding many *Instruments* to a :class:`~rateslib.instruments.Portfolio`, provided those
    *Instruments* have been properly configured, allows their fixing exposures to be
    analysed and aggregated. *Instruments* with no fixing exposures, such as a *FixedRateBond*,
    will simply be ignored.

    .. ipython:: python

       frb = FixedRateBond(dt(2000, 1, 3), "10y", fixed_rate=2.5, spec="us_gb")
       pf = Portfolio([irs, sbs, fra, xcs, frn, stir, iirs, zcs, frb])
       pf.fixings_table(approximate=True, right=dt(2000, 8, 1))
