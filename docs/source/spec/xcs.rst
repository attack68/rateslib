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

JPY
*****

.. _spec-jpyusd-xcs:

JPY/USD
---------

.. ipython:: python

   defaults.spec["jpyusd_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="jpyusd_xcs").kwargs

AUD
*****

AUD/USD (AONIA)
---------------------

.. _spec-audusd-xcs:

.. ipython:: python

   defaults.spec["audusd_xcs"]
   XCS(dt(2000, 1, 1), "10y", spec="audusd_xcs").kwargs

AUD/USD (BBSW 3m)
---------------------

.. _spec-audusd-xcs3:

.. ipython:: python

   defaults.spec["audusd_xcs3"]
   XCS(dt(2000, 1, 1), "10y", spec="audusd_xcs3").kwargs

NZD
*****

NZD/USD (NFix 3m)
-------------------

.. _spec-nzdusd-xcs3:

.. ipython:: python

   defaults.spec["nzdusd_xcs3"]
   XCS(dt(2000, 1, 1), "10y", spec="nzdusd_xcs3").kwargs

NZD/AUD (NFix 3m/BBSW 3m)
--------------------------

.. _spec-nzdaud-xcs3:

.. ipython:: python

   defaults.spec["nzdaud_xcs3"]
   XCS(dt(2000, 1, 1), "10y", spec="nzdaud_xcs3").kwargs
