
*****
Bill
*****

USD
********

.. _spec-usd-gbb:

Government Bill
----------------

Aliases: **"ustb"**

.. ipython:: python
   :suppress:

   from rateslib import *

.. ipython:: python

   defaults.spec["usd_gbb"]
   Bill(dt(2000, 1, 1), "3m", spec="usd_gbb").kwargs
