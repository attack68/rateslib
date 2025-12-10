
*********
FXOption
*********

.. ipython:: python
   :suppress:

   from rateslib import *

EURUSD
********

.. _spec-eurusd-call:

.. ipython:: python

   defaults.spec["eurusd_call"]
   from rateslib.instruments import FXCall
   FXCall(eval_date=dt(2000, 1, 1), expiry="3m", strike=1.10, spec="eurusd_call").kwargs
