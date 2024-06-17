.. _spec-sbs:

.. ipython:: python
   :suppress:

   from rateslib import *

****
SBS
****

EUR
********

.. _spec-eur-sbs36:

3s6s
----------

.. ipython:: python

   defaults.spec["eur_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="eur_sbs36").kwargs


NOK
*****

.. _spec-nok-sbs36:

3s6s
-----

.. ipython:: python

   defaults.spec["nok_sbs36"]
   SBS(dt(2000, 1, 1), "10y", spec="nok_sbs36").kwargs
