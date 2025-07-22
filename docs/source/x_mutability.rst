.. _mutability-doc:

******************
Mutability
******************

General guidance in *rateslib* is to create objects and **not** attempt to mutate them directly,
either by overwriting or creating class attributes. For example, don't create a
:class:`~rateslib.instruments.FixedRateBond` and then attempt to *change* that bond's
``effective``, or ``termination`` date or ``frequency``. *Instead*, just create a new instance
with the relevant modifications to input arguments.

However, *Rateslib* does not specifically place mutability
guards on objects and adopts the general philosophy of Python's flexible nature.

Certain objects are defined as *mutable by updates*, and these have dedicated methods
to allow the types of mutations users will typically want to perform.

Mutable by update
******************

There are the following key objects with *update* methods, and defined as being *mutable by update*.

- :class:`~rateslib.curves.Curve`
- :class:`~rateslib.curves.LineCurve`
- :class:`~rateslib.fx_volatility.FXDeltaVolSmile`
- :class:`~rateslib.fx_volatility.FXSabrSmile`
- :class:`~rateslib.fx.FXRates`

The first four of these are designed to be directly mutated by a :class:`~rateslib.solver.Solver`.
This means that the *Solver* will overwrite the objects' values with its own updates.
The *Solver's* updates will also utilise AD and the variables associated with the *Solver*,
meaning it will destroy any user input *variables* when it runs an *iteration*.

The latter in this list also defines its own AD and *variables* and will ignore user *variables*.

.. warning::

   **Exogneous** :class:`~rateslib.dual.Variable` should not be used with these objects due to
   their stated nature of being mutated and overwritten by the *Solver* and or its internals.

:ref:`Splines <splines-doc>` are also mutable with a dedicated
:meth:`~rateslib.splines.PPSplineF64.csolve` method to calibrate them to datapoints.

Mutable by association
***********************

The following objects are defined as *mutable by association*, since they function as containers
for *mutable by update* objects, or in the case of the latter are derived from a container.

- :class:`~rateslib.curves.CompositeCurve`
- :class:`~rateslib.curves.MultiCsaCurve`
- :class:`~rateslib.fx_volatility.FXDeltaVolSurface`
- :class:`~rateslib.fx_volatility.FXSabrSurface`
- :class:`~rateslib.fx.FXForwards`
- :class:`~rateslib.curves.ProxyCurve`

The only one of these objects that contains an *update* method is
:meth:`FXForwards.update <rateslib.fx.FXForwards.update>`, and only to be backwards
compatible when state management was not automatic in earlier versions of *rateslib*.
This method offers the convenience
of updating multiple *FXRates* objects via a single call, but it is no longer necessary.

Object states and the cache
****************************

Internally, objects maintain a record of their **state**, and may also keep a **cache**.

.. ipython:: python

   curve = Curve({dt(2025, 1, 1): 1.0, dt(2026, 1, 1): 0.97})
   _ = (curve[dt(2025, 2, 1)], curve[dt(2025, 8, 1)])
   curve._state
   curve._cache

When *officially* updated, their *state* will change and this will also clear the *cache*.

.. ipython:: python

   curve.update_node(dt(2026, 1, 1), 0.98)
   curve._state
   curve._cache

When methods on *mutable by association* objects are called they will perform a *validation*,
and update themselves if they detect one of their contained objects has changed state,
to ensure that erroneous results do not feed through.

.. ipython:: python

   fxr = FXRates({"eurusd": 1.10}, settlement=dt(2025, 1, 5))
   fxf = FXForwards(fx_rates=fxr, fx_curves={"eureur": curve, "usdusd": curve, "eurusd": curve})
   fxf.rate("eurusd", dt(2025, 2, 1))

   fxr.update({"eurusd": 1.20})  #  <-  the FXRates object is updated
   fxf.rate("eurusd", dt(2025, 2, 1))  #  <-  should auto-detect the new state

Immutables
***********

Objects such as *Calendars* (:class:`~rateslib.scheduling.Cal`,
:class:`~rateslib.scheduling.UnionCal`, :class:`~rateslib.scheduling.NamedCal`) are considered
immutable, as well *Number* types (:class:`~rateslib.dual.Dual`, :class:`~rateslib.dual.Dual2`,
:class:`~rateslib.dual.Variable`) and a :class:`~rateslib.scheduling.Schedule`.

Instruments
************

Instances of *Instruments*, *Legs* and *Periods* should not be considered user mutable.

Internally, they do contain routines for setting mid-market prices on *unpriced varieties*. For
example an :class:`~rateslib.instruments.IRS`, which has no ``fixed_rate`` set at initialisation,
or an :class:`~rateslib.instruments.FXCall`, whose strike is indefinitely set as a delta-% at
initialisation, will have its parameters definitively attributed for pricing and risk. These
changes are automatically controlled.

Solver safeguards
******************

The :class:`~rateslib.solver.Solver` is a central component used for pricing and risk calculation.
It also keeps track of the state of the objects within its scope, since without doing so
errors may be inadvertently introduced.

Two examples are shown below. The **first example** updates the same *Curve* with different
*Solvers* and demonstrates the generated error message.

.. ipython:: python

   curve = Curve({dt(2025, 1, 1): 1.0, dt(2026, 1, 1): 1.0})
   solver1 = Solver(curves=[curve], instruments=[IRS(dt(2025, 1, 1), "1m", spec="usd_irs", curves=curve)], s=[1.0])
   solver2 = Solver(curves=[curve], instruments=[IRS(dt(2025, 1, 1), "1m", spec="usd_irs", curves=curve)], s=[5.0])

   # solver2 has updated the curve after solver1 did. Try to price with solver1...
   try:
       IRS(dt(2025, 1, 1), "2m", spec="usd_irs", curves=curve).rate(solver=solver1)
   except ValueError as e:
       print(e)

In this **second example** a user calls an update method and adjusts some market data (or
perhaps directly mutates a *Curve*) but does not reiterate the *Solver*.

.. ipython:: python

   curve = Curve({dt(2025, 1, 1): 1.0, dt(2026, 1, 1): 1.0})
   fxr = FXRates({"eurusd": 1.10}, settlement=dt(2025, 1, 5))
   fxf = FXForwards(fx_rates=fxr, fx_curves={"eureur": curve, "usdusd": curve, "eurusd": curve})
   solver1 = Solver(curves=[curve], instruments=[IRS(dt(2025, 1, 1), "1m", spec="usd_irs", curves=curve)], s=[1.0], fx=fxf)

   # user updates the FXrates
   fxr.update({"eurusd": 1.20})

   # Try to price with solver1...
   import warnings
   with warnings.catch_warnings(record=True) as w:
       IRS(dt(2025, 1, 1), "2m", spec="usd_irs", curves=curve).rate(solver=solver1)
       print(w[-1].message)
