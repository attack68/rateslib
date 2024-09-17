.. _spec-stir:

.. ipython:: python
   :suppress:

   from rateslib import *

*************
STIR Futures
*************

EUR
********

.. _spec-eur-stir:

ESTR 3m
----------

Aliases: **"estr3mf"**.

.. ipython:: python

   defaults.spec["eur_stir"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 6, 21), spec="eur_stir").kwargs

.. _spec-eur-stir1:

ESTR 1m Averaged
------------------

Aliases: **"estr1mf"**.

.. ipython:: python

   defaults.spec["eur_stir1"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 4, 19), spec="eur_stir1").kwargs

.. _spec-eur-stir3:

Euribor 3m
-----------

Aliases: **"euribor3mf"**.

.. ipython:: python

   defaults.spec["eur_stir3"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 6, 21), spec="eur_stir3").kwargs


GBP
**********

Aliases: **"sonia3mf"**.

.. _spec-gbp-stir:

SONIA 3m
---------

.. ipython:: python

   defaults.spec["gbp_stir"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 6, 21), spec="gbp_stir").kwargs


USD
*******

Aliases: **"sofr3mf"**.

.. _spec-usd-stir:

SOFR 3m
---------

.. ipython:: python

   defaults.spec["usd_stir"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 6, 21), spec="usd_stir").kwargs

.. _spec-usd-stir1:

SOFR 1m Averaged
-----------------

Aliases: **"sofr1mf"**.

.. ipython:: python

   defaults.spec["usd_stir1"]
   STIRFuture(dt(2023, 3, 15), dt(2023, 4, 19), spec="usd_stir1").kwargs
