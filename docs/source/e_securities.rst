.. _securities-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Securities
**********************

Securities are generally one-leg instruments which have
been packaged to provide specific methods relevant to their
nature. For example bonds have yield-to-maturity and accrued interest
for example. Each class provides further documentation on those specifics.

.. autosummary::
   rateslib.instruments.FixedRateBond
   rateslib.instruments.FloatRateNote
   rateslib.instruments.Bill
   rateslib.instruments.IndexFixedRateBond
   rateslib.instruments.BondFuture
   rateslib.instruments._BaseBondInstrument
