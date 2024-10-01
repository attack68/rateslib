
*****
Bill
*****

.. ipython:: python
   :suppress:

   from rateslib import *

USD
********

.. _spec-usd-gbb:

Government Bill
----------------

Aliases: **"ustb"**

.. ipython:: python

   defaults.spec["us_gbb"]
   from rateslib.instruments.bonds import US_GBB
   US_GBB.kwargs
   Bill(dt(2000, 1, 1), "3m", spec="us_gbb").kwargs

GBP
********

.. _spec-uk-gbb:

Government Bill
----------------

Aliases: **"uktb"**

.. ipython:: python

   defaults.spec["uk_gbb"]
   from rateslib.instruments.bonds import UK_GBB
   UK_GBB.kwargs
   Bill(dt(2000, 1, 1), "3m", spec="uk_gbb").kwargs

SEK
********

.. _spec-se-gbb:

Government Bill
----------------

Aliases: **"sgbb"**

.. ipython:: python

   defaults.spec["se_gbb"]
   from rateslib.instruments.bonds import SE_GBB
   SE_GBB.kwargs
   Bill(dt(2000, 1, 1), "3m", spec="se_gbb").kwargs
