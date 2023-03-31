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

The ``rateslib.legs`` module creates ``Legs`` which
essentially contain a list of :ref:`Periods<periods-doc>`. The pricing, and
risk, calculations of ``Legs`` resolves to a linear sum of those same calculations
looped over all of the individual :ref:`Periods<periods-doc>`.
Like :ref:`Periods<periods-doc>` it is probably quite
rare that ``Legs`` will be instantiated directly, rather they will form the
components of :ref:`Instruments<instruments-doc>`, but none-the-less, this page
describes their construction.

The
following ``Legs`` are provided:

.. autosummary::
   rateslib.legs.BaseLeg
   rateslib.legs.FixedLeg
   rateslib.legs.FloatLeg
   rateslib.legs.BaseLegExchange
   rateslib.legs.FixedLegExchange
   rateslib.legs.FloatLegExchange
   rateslib.legs.FixedLegExchangeMtm
   rateslib.legs.FloatLegExchangeMtm
   rateslib.legs.CustomLeg

``Legs``, similar to ``Periods``, are defined as having the following the methods:

.. autosummary::
   rateslib.legs.BaseLeg.npv
   rateslib.legs.BaseLeg.analytic_delta
   rateslib.legs.BaseLeg.cashflows

Basic Leg Inputs
----------------
The :class:`~rateslib.legs.BaseLeg` is an abstract base class providing the shared
input arguments used by all ``Leg`` types. Besides ``fixed_rate``, a
:class:`~rateslib.legs.FixedLeg` can demonstrate all of the standard arguments to
a :class:`~rateslib.legs.BaseLeg`.

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
       fixings=None,
       fixing_method="rfr_payment_delay",
       method_param=None,
       spread_compound_method="none_simple",
   )
   float_leg.cashflows(curve)

The basic ``Legs`` are most commonly used in the construction
of :class:`~rateslib.instruments.IRS` and :class:`~rateslib.instruments.SBS`.

Legs with Exchanged Notionals
-----------------------------

``Bonds`` and ``CrossCurrencySwaps`` involve ``Legs`` with exchanged
notionals, which are represented as :class:`~rateslib.periods.Cashflow` s.
Notionals are always exchanged at the end on these ``Legs`` with
the option of also being exchanged at the start too.

The arguments are the same as the previous :class:`~rateslib.legs.FixedLeg`
and :class:`~rateslib.legs.FloatLeg` classes, except there are now the
additional arguments ``initial_exchange`` and ``payment_lag_exchange``.
The ``payment_lag_exchange`` argument allows to configure separate lags
for notional exchanges and regular period flows, which is common practice
on ``CrossCurrencySwaps`` for example.

.. ipython:: python

   fixed_leg_exch = FixedLegExchange(
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
       payment_lag_exchange=0,
   )
   fixed_leg_exch.cashflows(curve)

Mark-to-Market Exchanged Legs
-----------------------------
MTM ``Legs`` are common on ``CrossCurrencySwaps``. They require
a domestic, ``alt_notional`` argument from which the leg ``notional`` is
derived based on ``fx_fixings`` that take place during the settlement
of the ``Leg``. Otherwise, the arguments are the same as the
previous :class:`~rateslib.legs.FixedLegExchange`
and :class:`~rateslib.legs.FloatLegExchange`.

.. ipython:: python

   float_leg_exch = FloatLegExchangeMtm(
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
