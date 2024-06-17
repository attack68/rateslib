
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

   defaults.spec["us_gbb"]
   Bill(dt(2000, 1, 1), "3m", spec="us_gbb").kwargs
