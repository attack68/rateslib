.. _spec-usd-irs:

********
USD IRS
********

Aliases: **"sofr"**.

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["usd_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="usd_irs").kwargs
