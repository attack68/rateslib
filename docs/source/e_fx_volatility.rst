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

Currently, in v1.2.x, there is no ability to build a volatility surface, but this is planned for
short term implementation. Volatility inputs to pricing formulae currently rely on an explicit
volatility value.

The following *Instruments* are currently available.

.. inheritance-diagram:: rateslib.instruments.FXCall rateslib.instruments.FXPut rateslib.instruments.FXRiskReversal rateslib.instruments.FXStraddle rateslib.instruments.FXStrangle
   :private-bases:
   :parts: 1

.. autosummary::
   rateslib.instruments.FXCall
   rateslib.instruments.FXPut
   rateslib.instruments.FXRiskReversal
   rateslib.instruments.FXStraddle
   rateslib.instruments.FXStrangle

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

Typing `EURUSD Curncy OVML` into Bloomberg will bring up the Bloomberg currency options pricer.
This can be replicated with *rateslib* native functionality.

.. container:: twocol

   .. container:: leftside40

      .. ipython:: python

         fxc = FXCall(
             pair="eurusd",
             expiry=dt(2023, 6, 16),
             notional=20e6,
             strike=1.101,
             payment_lag=dt(2023, 3, 20),
             delivery_lag=2,
             calendar="tgt",
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

   .. container:: rightside60

      .. image:: _static/fx_opt_bbg_eurusd.png
          :alt: Bloomberg EURUSD option pricer
          :width: 400

.. raw:: html

   <div class="clear"></div>

The *Call* option priced above is partly unpriced becuase the premium is not
directly specified. This means that *rateslib* will always assert the premium
to be mid-market, based on the prevailing *Curves*, *FXForwards* and *vol* parameters
supplied.

Changing some of the pricing parameters provides different prices. *Rateslib* is
compared to Bloomberg's OVML.

.. list-table::
   :widths: 20 10 10 10 10 10 10 10 10
   :header-rows: 3

   * - Premium currency:
     - usd
     - usd
     - usd
     - usd
     - eur
     - eur
     - eur
     - eur
   * - Premium date:
     - 20/3/23
     - 20/3/23
     - 20/6/23
     - 20/6/23
     - 20/3/23
     - 20/3/23
     - 20/6/23
     - 20/6/23
   * - Delta type:
     - Spot
     - Forward
     - Spot
     - Forward
     - Spot (pa)
     - Forward (pa)
     - Spot (pa)
     - Forward (pa)
   * - Option rate (*rateslib*):
     - 69.3783
     - 69.3783
     - 70.2258
     - 70.2258
     - 0.65359
     - 0.65359
     - 0.65785
     - 0.65785
   * - Option rate (BBG):
     - 69.378
     - 69.378
     - 70.226
     - 70.226
     - 0.6536
     - 0.6536
     - 0.6578
     - 0.6578
   * - Delta % (*rateslib*):
     - 0.25012
     - 0.25175
     - 0.25012
     - 0.25175
     - 0.24359
     - 0.24518
     - 0.24359
     - 0.24518
   * - Delta % (BBG):
     - 0.25012
     - 0.25175
     - 0.25013
     - 0.25176
     - 0.24359
     - 0.24518
     - 0.24355
     - 0.24518

Restrictions
-------------

*Rateslib* currently allows the `currency` of the `premium` to **only be either** the domestic
(LHS) or the foreign (RHS) currency of the FX pair of the option (which is also the default
if none is specified).

If the currency is specified as foreign, then the pricing metric will
be stated in **pips** and the percent delta calculations are unadjusted.

If the currency is stated as domestic, then the pricing metric is stated as
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
       strike="25d",
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxc.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd","usd")],
       fx=fxf,
       vol=8.9
   )
   fxc.analytic_greeks(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=8.9
   )["delta"]

When the pricing functions are called the **strike is implied** and automatically set on the
attached *FXCallPeriod*.

.. ipython:: python

   fxc.periods[0].strike

If the pricing parameters change the *Option* strike will adapt accordingly to maintain the
25% spot delta calculation.

.. ipython:: python

   fxc.rate(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd","usd")],
       fx=fxf,
       vol=10.0
   )
   fxc.analytic_greeks(
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=10.0
   )["delta"]
   fxc.periods[0].strike


Risk Reversals
================

:class:`~rateslib.instruments.FXRiskReversal` are included as a direct product
because they are frequently traded products and *Instruments* often used
in calibrating a volatility surface.

*RiskReversals* need to be specified by two different ``strike`` values; a
lower and a higher strike. These can be entered in delta terms. Pricing also allows
two different ``vol`` inputs in the absense of a volatility surface.

.. ipython:: python

   fxrr = FXRiskReversal(
       pair="eurusd",
       expiry=dt(2023, 6, 16),
       notional=20e6,
       strike=("-25d", "25d"),
       payment_lag=2,
       delivery_lag=2,
       calendar="tgt",
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
       calendar="tgt",
       premium_ccy="usd",
       delta_type="spot",
   )
   fxrr.plot_payoff(
       range=[1.025, 1.11],
       curves=[None, fxf.curve("eur", "usd"), None, fxf.curve("usd", "usd")],
       fx=fxf,
       vol=[10.15, 8.9],
   )
