.. _instruments-doc:

.. ipython:: python
   :suppress:

   from rateslib.instruments import *
   from datetime import datetime as dt

**********************
Instruments
**********************

``Instruments`` are compositions of :ref:`Legs<legs-doc>` or other
``Instruments``. The following ``Instruments`` are provided:

Securities
**********

.. autosummary::
   rateslib.instruments.FixedRateBond
   rateslib.instruments.FloatRateBond
   rateslib.instruments.Bill

Single Currency Derivatives
***************************

.. autosummary::
   rateslib.instruments.BaseDerivative
   rateslib.instruments.IRS
   rateslib.instruments.Swap
   rateslib.instruments.SBS
   rateslib.instruments.FRA

Multi-Currency Derivatives
**************************

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


Combinations and Utilities
**************************

.. autosummary::
   rateslib.instruments.Value
   rateslib.instruments.Spread
   rateslib.instruments.Fly


Examples
********

.. ipython:: python

   irs = IRS(dt(2022, 1, 1), "9M", "Q", fixed_rate=3.00, convention="30360", leg2_convention="Act360")
   irs.npv(curve)
   irs.spread(curve)
   irs.rate(curve)
   # set the fixed rate equal to mid-market and check NPV
   irs.fixed_rate = irs.rate(curve).real
   irs.npv(curve)
   irs.cashflows(curve)


..  .. autoclass:: rateslib.instruments.Value
          :members:
    .. autoclass:: rateslib.instruments.FixedRateBond
          :members:
    .. autoclass:: rateslib.instruments.FloatRateBond
          :members:
    .. autoclass:: rateslib.instruments.BaseDerivative
          :members:
    .. automethod:: rateslib.instruments.forward_fx
    .. autoclass:: rateslib.instruments.IRS
          :members: rate, spread, cashflows, npv
    .. autoclass:: rateslib.instruments.Swap
    .. autoclass:: rateslib.instruments.SBS
          :members: rate, spread, cashflows, npv
    .. autoclass:: rateslib.instruments.FXSwap
          :members: rate, cashflows, npv
    .. autoclass:: rateslib.instruments.NonMtmXCS
    .. autoclass:: rateslib.instruments.Spread
    .. autoclass:: rateslib.instruments.Fly
