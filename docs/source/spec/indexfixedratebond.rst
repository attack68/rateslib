.. ipython:: python
   :suppress:

   from rateslib import *

*******************
IndexFixedRateBond
*******************

USD
******

.. _spec-us-gbi:

Government Bonds
-------------------

Similar to *'us_gb'* with 3 month index lag and daily interpolation.

.. ipython:: python

   defaults.spec["us_gbi"]
   from rateslib.instruments.bonds.conventions import US_GB
   US_GB.kwargs
   IndexFixedRateBond(dt(2000, 1, 1), "10y", spec="us_gbi", fixed_rate=2.5).kwargs


GBP
********

.. _spec-uk-gbi:

Government Bonds
-----------------

Similar to *'uk_gb'* with 3 month index lag and daily interpolation.

.. ipython:: python

   defaults.spec["uk_gbi"]
   from rateslib.instruments.bonds.conventions import UK_GB
   UK_GB.kwargs
   IndexFixedRateBond(dt(2000, 1, 1), "10y", spec="uk_gbi", fixed_rate=2.5).kwargs
