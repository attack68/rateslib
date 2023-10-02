.. _singlecurrency-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

****************************
Single Currency Derivatives
****************************

Single currency derivatives are examples of the simplest two-leg
structures.

.. inheritance-diagram:: rateslib.instruments.IRS rateslib.instruments.FRA rateslib.instruments.SBS rateslib.instruments.ZCIS rateslib.instruments.ZCS rateslib.instruments.IIRS rateslib.instruments.STIRFuture
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.BaseDerivative
   rateslib.instruments.IRS
   rateslib.instruments.SBS
   rateslib.instruments.FRA
   rateslib.instruments.ZCS
   rateslib.instruments.ZCIS
   rateslib.instruments.IIRS
   rateslib.instruments.STIRFuture
