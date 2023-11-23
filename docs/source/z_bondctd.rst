.. _bondctd-doc:

.. ipython:: python
   :suppress:

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context

Bond Future CTD Multi-Scenario Analysis
********************************************

In late 2023 CTD analysis of the US 30Y Treasury Bond Future was worth exploring because
quickly rising yields led to multiple changes in the CTD bond.

This page will demonstrate how one might use *rateslib* to perform some of this analysis.

First we need to configure all of the *Instruments* and their prices. This is shown statically below.

.. ipython:: python

   data = DataFrame(
       data= [
           [FixedRateBond(dt(2022, 1, 1), dt(2039, 8, 15), fixed_rate=4.5, spec="ust", curves="bcurve"), 98.6641],
           [FixedRateBond(dt(2022, 1, 1), dt(2040, 2, 15), fixed_rate=4.625, spec="ust", curves="bcurve"), 99.8203],
           [FixedRateBond(dt(2022, 1, 1), dt(2041, 2, 15), fixed_rate=4.75, spec="ust", curves="bcurve"), 100.7734],
           [FixedRateBond(dt(2022, 1, 1), dt(2040, 5, 15), fixed_rate=4.375, spec="ust", curves="bcurve"), 96.6953],
           [FixedRateBond(dt(2022, 1, 1), dt(2039, 11, 15), fixed_rate=4.375, spec="ust", curves="bcurve"), 97.0781],
           [FixedRateBond(dt(2022, 1, 1), dt(2040, 11, 15), fixed_rate=4.25, spec="ust", curves="bcurve"), 94.8516],
           [FixedRateBond(dt(2022, 1, 1), dt(2039, 5, 15), fixed_rate=4.25, spec="ust", curves="bcurve"), 96.0469],
           [FixedRateBond(dt(2022, 1, 1), dt(2041, 5, 15), fixed_rate=4.375, spec="ust", curves="bcurve"), 96.1250],
           [FixedRateBond(dt(2022, 1, 1), dt(2040, 8, 15), fixed_rate=3.875, spec="ust", curves="bcurve"), 90.5938],
           [FixedRateBond(dt(2022, 1, 1), dt(2042, 11, 15), fixed_rate=4.00, spec="ust", curves="bcurve"), 90.4766],
           [FixedRateBond(dt(2022, 1, 1), dt(2043, 2, 15), fixed_rate=3.875, spec="ust", curves="bcurve"), 88.7656],
           [FixedRateBond(dt(2022, 1, 1), dt(2043, 8, 15), fixed_rate=4.375, spec="ust", curves="bcurve"),  95.0703],
           [FixedRateBond(dt(2022, 1, 1), dt(2042, 8, 15), fixed_rate=3.375, spec="ust", curves="bcurve"), 82.7188],
           [FixedRateBond(dt(2022, 1, 1), dt(2041, 8, 15), fixed_rate=3.75, spec="ust", curves="bcurve"), 88.4766],
           [FixedRateBond(dt(2022, 1, 1), dt(2042, 5, 15), fixed_rate=3.25, spec="ust", curves="bcurve"), 81.3828],
           [FixedRateBond(dt(2022, 1, 1), dt(2039, 2, 15), fixed_rate=3.50, spec="ust", curves="bcurve"), 88.1406],
       ],
       columns=["bonds", "prices"],
   )
   usz3 = BondFuture(  # Construct the BondFuture Instrument
       coupon=6.0,
       delivery=(dt(2023, 12, 1), dt(2023, 12, 29)),
       basket=data["bonds"],
       nominal=100e3,
       calendar="nyc",
       currency="usd",
       calc_mode="ust_long",
   )
   dlv = usz3.dlv(  # Analyse the deliverables as of the current prices
       future_price=115.9688,
       prices=data["prices"],
       settlement=dt(2023, 11, 22),
       repo_rate=5.413,
       convention="act360",
   )
   with option_context("display.float_format", lambda x: '%.6f' % x):
       print(dlv)

Compared with the Bloomberg read out for the same data:

.. image:: _static/usz3dlv.png
  :alt: Bloomberg DLV function
