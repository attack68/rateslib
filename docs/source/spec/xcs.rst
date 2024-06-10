.. _spec-xcs:

.. ipython:: python
   :suppress:

   from rateslib import *

****
XCS
****

EUR
********

.. _spec-eurusd-xcs:

EUR/USD
----------

.. ipython:: python

   defaults.spec["eurusd_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="eurusd_xcs").kwargs

.. _spec-eurgbp-xcs:

EUR/GBP
----------

.. ipython:: python

   defaults.spec["eurgbp_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="eurgbp_xcs").kwargs


GBP
**********

.. _spec-gbpusd-xcs:

GBP/USD
---------

.. ipython:: python

   defaults.spec["gbpusd_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="gbpusd_xcs").kwargs

.. _spec-gbpeur-xcs:

GBP/EUR
---------

.. ipython:: python

   defaults.spec["gbpeur_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="gbpeur_xcs").kwargs
