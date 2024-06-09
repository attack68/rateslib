.. _spec-eur-irs:

********
EUR IRS
********

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["eur_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs").kwargs