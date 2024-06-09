.. _spec-usd-gb:

********
USD GB
********

Aliases: **"ust"**

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["usd_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="usd_gb", fixed_rate=2.5).kwargs
