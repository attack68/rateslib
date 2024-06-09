.. _spec-irs:

.. ipython:: python
   :suppress:

   from rateslib import *

****
IRS
****

USD
********

.. _spec-usd-irs:

SOFR
-----

Aliases: **"sofr"**.

.. ipython:: python

   defaults.spec["usd_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="usd_irs").kwargs


EUR
*****

.. _spec-eur-irs:

ESTR
-----

.. ipython:: python

   defaults.spec["eur_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs").kwargs

.. _spec-eur-irs3:

3m Euribor
-------------

.. ipython:: python

   defaults.spec["eur_irs3"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs3").kwargs

.. _spec-eur-irs6:

6m Euribor
-----------

.. ipython:: python

   defaults.spec["eur_irs6"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs6").kwargs


GBP
*****

.. _spec-gbp-irs:

SONIA
-----

.. ipython:: python

   defaults.spec["gbp_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="gbp_irs").kwargs

CHF
*****

.. _spec-chf-irs:

SARON
-----

.. ipython:: python

   defaults.spec["chf_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="chf_irs").kwargs
