.. _spec-cds:

.. ipython:: python
   :suppress:

   from rateslib import *

****
CDS
****

US
********

.. _spec-us-ig-cds:

Investment Grade
------------------

.. ipython:: python

   defaults.spec["us_ig_cds"]
   CDS(dt(2000, 12, 20), "10y", spec="us_ig_cds").kwargs
