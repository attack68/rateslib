.. _multicurrency-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

****************************
Multi-Currency Derivatives
****************************

Multi-currency derivatives are generally more complicated two-leg
structures.

.. autosummary::
   rateslib.instruments.BaseXCS
   rateslib.instruments.XCS
   rateslib.instruments.FixedFloatXCS
   rateslib.instruments.FloatFixedXCS
   rateslib.instruments.FixedFixedXCS
   rateslib.instruments.NonMtmXCS
   rateslib.instruments.NonMtmFixedFloatXCS
   rateslib.instruments.NonMtmFixedFixedXCS
   rateslib.instruments.forward_fx
