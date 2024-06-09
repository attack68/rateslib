.. _spec-eur-irs3:

********
EUR IRS3
********

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["eur_irs3"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs3").kwargs