.. _volatility-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

****************************
FX Volatility
****************************

Single currency derivatives are examples of the simplest two-leg
structures.

.. inheritance-diagram:: rateslib.instruments.FXCall rateslib.instruments.FXPut rateslib.instruments.FXRiskReversal
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.FXCall
   rateslib.instruments.FXPut
   rateslib.instruments.FXRiskReversal

FX options using the **Black 76 log-normal model** can be priced in *rateslib*.

FXForwards Market
==================

First, it is practical to have *Curves* and an :class:`~rateslib.fx.FXForwards` object determined
from non-volatility markets. See :ref:`FX forwards <fxf-doc>`. This will be used to forecast
forward FX rates relevant to the pricing of an arbitrary FX option.

.. ipython:: python

    # FXForwards for FXOptions
    eureur = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.9851909811629752}, calendar="tgt", id="eureur"
    )
    usdusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.976009366603271}, calendar="nyc", id="usdusd"
    )
    eurusd = Curve(
        {dt(2023, 3, 16): 1.0, dt(2023, 9, 16): 0.987092591908283}, id="eurusd"
    )
    fxr = FXRates({"eurusd": 1.0615}, settlement=dt(2023, 3, 20))
    fxf = FXForwards(
        fx_curves={"eureur": eureur, "eurusd": eurusd, "usdusd": usdusd},
        fx_rates=fxr
    )
    fxf._set_ad_order(1)
    fxf.swap("eurusd", [dt(2023, 3, 20), dt(2023, 6, 20)])  # should be 60.1 points

FXCall Option
==============

Next define an :class:`~rateslib.fx.FXCall` option. Below we have used the *NoInput* argument
for some arguments which can otherwise be ignored.

.. ipython:: python

   fxc = FXCall(
       pair="eurusd",
       expiry=dt(2023, 6, 16),   # a specified 3M expiry
       notional=20e6,            # 1mm EUR
       strike=1.101,
       payment_lag=2,            # premium is paid 2 days after expiry
       delivery_lag=2,           # FX rate is considered as spot from expiry
       calendar="tgt,nyc",       # used to determine spot and delivery from expiry
       modifier="f",             # modify delivery and payment forward
       premium_ccy="usd",
       eval_date=NoInput(0),    # expiry is not a string tenor
       premium=NoInput(0),      # option is left unpriced
       option_fixing=NoInput(0),  # option not yet fixed
       delta_type=NoInput(0),   # strike is set explicitly (not as %delta)
       curves=NoInput(0),       # pricing will be dynamic
       spec=NoInput(0),         # parameters set explicitly
   )

Get the price of the *FXCall* option in premium terms

.. ipython:: python

   fxc.rate(curves=[eurusd, usdusd], fx=fxf, vol=0.089)
