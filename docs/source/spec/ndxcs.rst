.. _spec-ndxcs:

.. ipython:: python
   :suppress:

   from rateslib import *

*****
NDXCS
*****

INR
********

.. _spec-inrusd-ndxcs:

INR/USD
----------

An INR Fixed to SOFR non-deliverable into USD.
INR *Leg* must be *Leg1*.

.. ipython:: python

   defaults.spec["inrusd_ndxcs"]
   NDXCS(dt(2000, 1, 1), "10y", spec="inrusd_ndxcs").kwargs
