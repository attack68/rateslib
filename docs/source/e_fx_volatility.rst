.. _fx-volatility-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

****************************
FX Volatility
****************************

Interbank standard conventions for quoting FX volatility products are quite varied.
None-the-less, *rateslib* provides the most common definitions and products, all priced using
the **Black-76** model.

There is an :class:`~rateslib.fx_volatility.FXDeltaVolSmile`, for options with consistent expiries,
and an :class:`~rateslib.fx_volatility.FXDeltaVolSurface`, for a more generalised pricing model.
The ability to input ``vol`` as an explicit numeric value in pricing methods also exists.

The following *Instruments* are currently available.

.. inheritance-diagram:: rateslib.instruments.FXCall rateslib.instruments.FXPut rateslib.instruments.FXRiskReversal rateslib.instruments.FXStraddle rateslib.instruments.FXStrangle rateslib.instruments.FXBrokerFly
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.FXCall
   rateslib.instruments.FXPut
   rateslib.instruments.FXRiskReversal
   rateslib.instruments.FXStraddle
   rateslib.instruments.FXStrangle
   rateslib.instruments.FXBrokerFly

FXForwards Market
==================

As multi-currency derivatives, *FX Options* rely on the existence of an
:class:`~rateslib.fx.FXForwards` object, which is usually determined
from non-volatility markets. See :ref:`FX forwards <fxf-doc>`. This will be used to forecast
forward FX rates relevant to the pricing of an arbitrary *FX Option*.

For the purpose of this user guide page, we create such a market below.

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

.. _build-option-doc:

Building and Pricing an Option
================================

*Calls* and *Puts* can be replicated with *rateslib* native
functionality via :class:`~rateslib.instruments.FXCall` and
:class:`~rateslib.instruments.FXPut`.


.. ipython:: python

   fxc = FXCall(
      pair="eurusd",
      expiry=dt(2023, 6, 16),
      notional=20e6,
      strike=1.101,
      payment_lag=dt(2023, 3, 20),
      delivery_lag=2,
      calendar="tgt|fed",
      modifier="mf",
      premium_ccy="usd",
      eval_date=NoInput(0),
      option_fixing=NoInput(0),
      premium=NoInput(0),
      delta_type="forward",
      curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd","usd")],
      spec=NoInput(0),
   )
   fxc.rate(fx=fxf, vol=8.9)
   fxc.analytic_greeks(vol=8.9, fx=fxf)


The *Call* option priced above is partly unpriced becuase the premium is not
directly specified. This means that *rateslib* will always assert the premium
to be mid-market, based on the prevailing *Curves*, *FXForwards* and *vol* parameters
supplied.

Restrictions
-------------

*Rateslib* currently allows the `currency` of the `premium` to **only be either** the domestic
(LHS) or the foreign (RHS) currency of the FX pair of the option (which is also the default
if none is specified).

If the currency is left as the default foreign RHS, then the pricing metric will
be stated in **pips** and the percent delta calculations are unadjusted.

If the currency is specifically stated as the LHS domestic, then the pricing metric used is
**percentage of notional** and the percent delta calculations are **premium adjusted**.

Strikes given in Delta terms
=============================

Commonly interbank *Instruments* are quoted in terms of delta values and the strikes are not
explicitly stated. Suppose building a *FXCall* with a specified 25% delta.

.. ipython:: python

   fxc = FXCall(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike="25d",  #  <-  Input for 25% delta
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxc.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd","usd")],
       fx=fxf,
       vol=8.9
   )

When pricing functions are called, the strike on the option is implied from the vol and the delta value. This may
require a root finding algorithm particularly if the ``vol`` is given as a *Smile* or a *Surface*. Relevant pricing
parameters can be seen by viewing :meth:`~rateslib.instruments.FXOption.analytic_greeks`. The strike is also
automatically assigned, temporarily, to the attached **FXCallPeriod**

.. ipython:: python

   fxc.analytic_greeks(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9
   )
   fxc.periods[0].strike

With altered pricing parameters, the *Option* strike will adapt accordingly to maintain the
25% spot delta calculation.

.. ipython:: python

   fxc.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd","usd")],
       fx=fxf,
       vol=10.0,   #  <- A different vol will imply a different strike to maintain the same delta
   )
   fxc.analytic_greeks(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=10.0
   )
   fxc.periods[0].strike

Straddles
==========

An :class:`~rateslib.instruments.FXStraddle` is the most frequently traded instrument for outright exposure to
volatility. *Straddles* are defined by a single strike, which can be a defined numeric value (for a 'struck' deal),
or an or associated value, e.g. "atm_delta", "atm_forward" or "atm_spot".

The default pricing ``metric`` for an *FX Straddle* is *'vol'* points.

.. ipython:: python

   fxstr = FXStraddle(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike="atm_delta",
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxstr.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9,
   )
   fxstr.analytic_greeks(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9,
   )
   fxstr.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9,
   )

.. plot::

   from rateslib.curves import Curve
   from rateslib.instruments import FXStraddle
   from rateslib import dt
   from rateslib.fx import FXForwards, FXRates

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
   fxrr = FXStraddle(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike="atm_delta",
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxrr.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9,
   )

Risk Reversals
================

:class:`~rateslib.instruments.FXRiskReversal` are frequently traded products and often used
in calibrating a volatility *Surface* or *Smile*.

*RiskReversals* need to be specified by two different ``strike`` values; a
lower and a higher strike. These can be entered in delta terms. Pricing also allows
two different ``vol`` inputs if a volatility *Surface* or *Smile* is not given.

The default pricing ``metric`` for a *RiskReversal* is *'vol'* which calculates the difference in volatility
quotations for each option.

.. ipython:: python

   fxrr = FXRiskReversal(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike=("-25d", "25d"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxrr.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[10.15, 8.9]
   )
   fxrr.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[10.15, 8.9]
   )

.. plot::

   from rateslib.curves import Curve
   from rateslib.instruments import FXRiskReversal
   from rateslib import dt
   from rateslib.fx import FXForwards, FXRates

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
   fxrr = FXRiskReversal(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike=("-25d", "25d"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxrr.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[10.15, 8.9],
   )

Strangles
==========

The other common *Instrument* combination for calibrating *Surfaces* and *Smiles* is an
:class:`~rateslib.instruments.FXStrangle`. Again, the strangle requires two strike inputs,
which can be input in delta terms or numeric value.

The default pricing ``metric`` for a strangle is *'single_vol'*, which quotes a single volatility
value used to price the strike and premium for each option. This is a complex calculation: *Rateslib* uses
an iteration to calculate this (see :meth:`~rateslib.instruments.FXStrangle.rate`) from a *Surface* or *Smile*.

.. ipython:: python

   fxstg = FXStrangle(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike=("-25d", "25d"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxstg.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[10.15, 8.9]
   )
   fxstg.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=9.533895,
   )

.. plot::

   from rateslib.curves import Curve
   from rateslib.instruments import FXStrangle
   from rateslib import dt
   from rateslib.fx import FXForwards, FXRates

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
   fxstg = FXStrangle(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike=("-25d", "25d"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxstg.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=9.533895,
   )


BrokerFly
==========

The final instrument commonly seen is an :class:`~rateslib.instruments.FXBrokerFly`.
This is a combination of an *FXStrangle* and an *FXStraddle*, where the ``notional`` on the
*FXStraddle* is determined at mid-market by making the structure *vega neutral*.

The default pricing ``metric`` is *'single_vol'* which calculates the single volatility price of the
*FXStrangle* and subtracts the volatility of the *FXStraddle*.

.. ipython:: python

   fxbf = FXBrokerFly(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=[20e6, -13.5e6],
       strike=(("-25d", "25d"), "atm_delta"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxbf.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[[10.15, 8.9], 7.5]
   )
   fxbf.plot_payoff(
       range=[1.000, 1.150],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=9.533895,
   )

.. plot::

   from rateslib.curves import Curve
   from rateslib.instruments import FXBrokerFly
   from rateslib import dt
   from rateslib.fx import FXForwards, FXRates

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
   fxbf = FXBrokerFly(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=[20e6, -13.5e6],
       strike=(("-25d", "25d"), "atm_delta"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt|fed",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxbf.plot_payoff(
       range=[1.000, 1.150],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=9.533895,
   )
