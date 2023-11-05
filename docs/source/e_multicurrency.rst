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


.. inheritance-diagram:: rateslib.instruments.XCS rateslib.instruments.FXSwap rateslib.instruments.FXExchange
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.XCS
   rateslib.instruments.FXSwap
   rateslib.instruments.FXExchange
   rateslib.instruments.forward_fx
