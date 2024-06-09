.. _spec-eur-irs6:

********
EUR IRS6
********

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["eur_irs6"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs6").kwargs