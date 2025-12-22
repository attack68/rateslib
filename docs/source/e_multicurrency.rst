.. _multicurrency-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

****************************
Multi-Currency Derivatives
****************************

Multi-currency derivatives are generally more complicated two-leg
structures. Note the *IRS* listed here allows *ND-IRS* directly.

.. autosummary::
   rateslib.instruments.XCS
   rateslib.instruments.FXSwap
   rateslib.instruments.FXForward
   rateslib.instruments.NDF
   rateslib.instruments.NDXCS
   rateslib.instruments.IRS
   rateslib.instruments._BaseInstrument

