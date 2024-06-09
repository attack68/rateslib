.. ipython:: python
   :suppress:

   from rateslib import *

**************
FixedRateBond
**************

USD
****

.. _spec-usd-gb:

Government Bonds
------------------

Aliases: **"ust"**

.. ipython:: python

   defaults.spec["usd_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="usd_gb", fixed_rate=2.5).kwargs


GBP
********

.. _spec-gbp-gb:

Government Bonds
-----------------

Aliases: **"ukt"** and **"gilt"**

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["gbp_gb"]
   FixedRateBond(dt(2000, 1, 1), "10y", spec="gbp_gb", fixed_rate=2.5).kwargs
