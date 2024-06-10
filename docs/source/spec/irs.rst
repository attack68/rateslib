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

.. _spec-eur-irs1:

1m Euribor
-----------

.. ipython:: python

   defaults.spec["eur_irs1"]
   IRS(dt(2000, 1, 1), "10y", spec="eur_irs1").kwargs

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

SEK
*****

.. _spec-sek-irs:

SWESTR
------

.. ipython:: python

   defaults.spec["sek_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="sek_irs").kwargs

.. _spec-sek-irs3:

3m Stibor
----------

.. ipython:: python

   defaults.spec["sek_irs3"]
   IRS(dt(2000, 1, 1), "10y", spec="sek_irs3").kwargs

NOK
*****

.. _spec-nok-irs:

NOWA
------

.. ipython:: python

   defaults.spec["nok_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="nok_irs").kwargs

.. _spec-nok-irs3:

3m Nibor
---------

.. ipython:: python

   defaults.spec["nok_irs3"]
   IRS(dt(2000, 1, 1), "10y", spec="nok_irs3").kwargs

.. _spec-nok-irs6:

6m Nibor
----------

.. ipython:: python

   defaults.spec["nok_irs6"]
   IRS(dt(2000, 1, 1), "10y", spec="nok_irs6").kwargs

CAD
*****

.. _spec-cad-irs:

CORRA
----------

.. ipython:: python

   defaults.spec["cad_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="cad_irs").kwargs

.. _spec-cad-irs-le-1y:

CORRA with tenor less than or equal to 1Y
-------------------------------------------

.. ipython:: python

   defaults.spec["cad_irs_le_1y"]
   IRS(dt(2000, 1, 1), "9m", spec="cad_irs_le_1y").kwargs

JPY
*****

.. _spec-jpy-irs:

TONA
----------

.. ipython:: python

   defaults.spec["jpy_irs"]
   IRS(dt(2000, 1, 1), "10y", spec="jpy_irs").kwargs
