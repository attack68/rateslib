.. _spec-gbp-gb:

********
GBP GB
********

Aliases: **"ukt"** and **"gilt"**

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["gbp_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="gbp_gb", fixed_rate=2.5).kwargs
