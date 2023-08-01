.. _legs-doc:

.. ipython:: python
   :suppress:

   from rateslib.periods import *
   from rateslib.legs import *
   from datetime import datetime as dt
   curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
   )

****
Legs
****

The ``rateslib.legs`` module creates *Legs* which
typically contain a list of :ref:`Periods<periods-doc>`. The pricing, and
risk, calculations of *Legs* resolves to a linear sum of those same calculations
looped over all of the individual *Periods*.
Like *Periods*, it is probably quite
rare that *Legs* will be instantiated directly, rather they will form the
components of :ref:`Instruments<instruments-doc>`, but none-the-less, this page
describes their construction.

The following *Legs* are provided, click on the links for a full description of each
*Leg* type:

.. autosummary::
   rateslib.legs.BaseLeg
   rateslib.legs.BaseLegMtm
   rateslib.legs.FixedLeg
   rateslib.legs.FloatLeg
   rateslib.legs.IndexFixedLeg
   rateslib.legs.ZeroFloatLeg
   rateslib.legs.ZeroFixedLeg
   rateslib.legs.ZeroIndexLeg
   rateslib.legs.FixedLegMtm
   rateslib.legs.FloatLegMtm
   rateslib.legs.CustomLeg

*Legs*, similar to *Periods*, are defined as having the following the methods:

.. autosummary::
   rateslib.legs.BaseLeg.npv
   rateslib.legs.BaseLeg.analytic_delta
   rateslib.legs.BaseLeg.cashflows

Basic Leg Inputs
----------------
The :class:`~rateslib.legs.BaseLeg` is an abstract base class providing the shared
input arguments used by all *Leg* types. Besides ``fixed_rate``, a
:class:`~rateslib.legs.FixedLeg` can demonstrate all of the standard arguments to
a *BaseLeg*.

For complete documentation of some of these inputs see :ref:`Scheduling<schedule-doc>`.

.. ipython:: python

   fixed_leg = FixedLeg(
       effective=dt(2022, 1, 15),
       termination=dt(2022, 12, 7),
       frequency="Q",
       stub=None,
       front_stub=dt(2022, 2, 28),
       back_stub=dt(2022, 11, 30),
       roll=None,
       eom=True,
       modifier="MF",
       calendar="nyc",
       payment_lag=2,
       notional=2000000,
       currency="usd",
       amortization=250000,
       convention="act360",
       fixed_rate=1.0,
       initial_exchange=False,
       final_exchange=False,
       payment_lag_exchange=0,
   )
   fixed_leg.cashflows(curve)

:class:`~rateslib.legs.FloatLeg` offer the same arguments with the additional
inputs that are appropriate for calculating a :class:`~rateslib.periods.FloatPeriod`.

.. ipython:: python

   float_leg = FloatLeg(
       effective=dt(2022, 1, 15),
       termination=dt(2022, 12, 7),
       frequency="Q",
       stub=None,
       front_stub=dt(2022, 2, 28),
       back_stub=dt(2022, 11, 30),
       roll=None,
       eom=True,
       modifier="MF",
       calendar="nyc",
       payment_lag=2,
       notional=2000000,
       currency="usd",
       amortization=250000,
       convention="act360",
       float_spread=1.0,
       initial_exchange=False,
       final_exchange=False,
       payment_lag_exchange=0,
       fixings=None,
       fixing_method="rfr_payment_delay",
       method_param=None,
       spread_compound_method="none_simple",
   )
   float_leg.cashflows(curve)

These basic *Legs* are most commonly used in the construction
of :class:`~rateslib.instruments.IRS` and :class:`~rateslib.instruments.SBS`.

Legs with Exchanged Notionals
-----------------------------

``Bonds``, ``CrossCurrencySwaps`` and ``IndexSwaps`` involve *Legs* with exchanged
notionals, which are represented as :class:`~rateslib.periods.Cashflow` s.
These ``Legs`` have the option of an initial exchange and also of a
final exchange. Interim exchanges (amortization) will be applied if
there is a final exchange.

The arguments are the same as the previous :class:`~rateslib.legs.FixedLeg`
and :class:`~rateslib.legs.FloatLeg` classes, except attention is drawn to the
provided arguments:

- ``initial_exchange``,
- ``final_exchange``,
- ``payment_lag_exchange``,

This allows for configuration of separate payment lags
for notional exchanges and regular period flows, which is common practice
on *CrossCurrencySwaps* for example.

.. ipython:: python

   fixed_leg_exch = FixedLeg(
       effective=dt(2022, 1, 15),
       termination=dt(2022, 7, 15),
       frequency="Q",
       stub=None,
       front_stub=None,
       back_stub=None,
       roll=None,
       eom=True,
       modifier="MF",
       calendar="nyc",
       payment_lag=2,
       notional=2000000,
       currency="usd",
       amortization=250000,
       convention="act360",
       fixed_rate=5.0,
       initial_exchange=True,
       final_exchange=True,
       payment_lag_exchange=0,
   )
   fixed_leg_exch.cashflows(curve)

Mark-to-Market Exchanged Legs
-----------------------------
``LegMtm`` objects are common on ``CrossCurrencySwaps``.
Whilst the other leg types are technically indifferent regarding the ``currency``
they are initialised with, *LegMtms* **require** a domestic currency and an alternative
currency against which MTM calculations can be measured. The ``notional`` of the
``MtmLeg`` is variable according to the fixed ``alt_notional`` and the forward
FX rates. Thus the additional arguments in this leg are:

- ``alt_notional``
- ``alt_currency``
- ``fx_fixings``
- ``notional`` is not used in this leg type and is overwritten.

Otherwise, the arguments are the same as the
previous :class:`~rateslib.legs.FixedLeg`
and :class:`~rateslib.legs.FloatLeg`.

.. ipython:: python

   float_leg_exch = FloatLegMtm(
       effective=dt(2022, 1, 3),
       termination=dt(2022, 7, 3),
       frequency="Q",
       stub=None,
       front_stub=None,
       back_stub=None,
       roll=None,
       eom=True,
       modifier="MF",
       calendar="nyc",
       payment_lag=2,
       notional=None,
       currency="usd",
       amortization=None,
       convention="act360",
       float_spread=0.0,
       fixings=None,
       fixing_method="rfr_payment_delay",
       method_param=None,
       spread_compound_method="none_simple",
       initial_exchange=True,
       final_exchange=True,
       payment_lag_exchange=0,
       alt_notional=2000000,
       alt_currency="eur",
       fx_fixings=None,
   )
   fxr = FXRates({"eurusd": 1.05}, settlement = dt(2022, 1, 3))
   fxf = FXForwards(fxr, {
       "usdusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.965}),
       "eureur": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.985}),
       "eurusd": Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.987}),
   })
   float_leg_exch.cashflows(curve, curve, fxf)
