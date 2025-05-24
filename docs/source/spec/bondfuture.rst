.. ipython:: python
   :suppress:

   from rateslib import *

**************
BondFuture
**************

USD
****

CME Treasury Futures
---------------------

.. _spec-us-gb-2y:

**2-year**

.. ipython:: python

   defaults.spec["us_gb_2y"]
   BondFuture(spec="us_gb_2y", delivery=(dt(2000, 3, 1), dt(2000, 3, 31))).kwargs

.. _spec-us-gb-3y:

**3-year**

.. ipython:: python

   defaults.spec["us_gb_3y"]
   BondFuture(spec="us_gb_3y", delivery=(dt(2000, 3, 1), dt(2000, 3, 31))).kwargs

.. _spec-us-gb-5y:

**5-year**

.. ipython:: python

   defaults.spec["us_gb_5y"]
   BondFuture(spec="us_gb_5y", delivery=(dt(2000, 3, 1), dt(2000, 3, 31))).kwargs

.. _spec-us-gb-10y:

**10-year**

.. ipython:: python

   defaults.spec["us_gb_10y"]
   BondFuture(spec="us_gb_10y", delivery=(dt(2000, 3, 1), dt(2000, 3, 31))).kwargs

.. _spec-us-gb-30y:

**30-year**

.. ipython:: python

   defaults.spec["us_gb_30y"]
   BondFuture(spec="us_gb_30y", delivery=(dt(2000, 3, 1), dt(2000, 3, 31))).kwargs

EUR
********

.. _spec-de-gb-2y:

**Eurex Schatz**

.. ipython:: python

   defaults.spec["de_gb_2y"]
   BondFuture(spec="de_gb_2y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-de-gb-5y:

**Eurex Bobl**

.. ipython:: python

   defaults.spec["de_gb_5y"]
   BondFuture(spec="de_gb_5y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-de-gb-10y:

**Eurex Bund**

.. ipython:: python

   defaults.spec["de_gb_10y"]
   BondFuture(spec="de_gb_10y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-de-gb-30y:

**Eurex Buxl**

.. ipython:: python

   defaults.spec["de_gb_30y"]
   BondFuture(spec="de_gb_30y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-fr-gb-5y:

**Eurex OAT 5y**

.. ipython:: python

   defaults.spec["fr_gb_5y"]
   BondFuture(spec="fr_gb_5y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-fr-gb-10y:

**Eurex OAT 10y**

.. ipython:: python

   defaults.spec["fr_gb_10y"]
   BondFuture(spec="fr_gb_10y", delivery=dt(2000, 3, 10)).kwargs

.. _spec-sp-gb-10y:

**Eurex BONO**

.. ipython:: python

   defaults.spec["sp_gb_10y"]
   BondFuture(spec="sp_gb_10y", delivery=dt(2000, 3, 10)).kwargs

CHF
********

.. _spec-ch-gb-10y:

**CONF Futures**

.. ipython:: python

   defaults.spec["ch_gb_10y"]
   BondFuture(spec="ch_gb_10y", delivery=dt(2000, 3, 10)).kwargs
