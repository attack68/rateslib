.. _mechanisms-doc:

******************
Pricing Mechanisms
******************

Summary
**************************

The pricing mechanisms in ``rateslib`` require ``Instruments`` and
``Curves``. ``fx`` objects (usually ``FXForwards``) may also be required
(for multi-currency instruments), and these
are all often interdependent and calibrated by a ``Solver``.

A careful API structure has been created which allows different mechanisms for
argument input to promote maximum flexibility which may be useful for certain
circumstances.

There are **three different modes of initialising an** ``Instrument``:

- **Without** specifying any pricing ``curves`` at initialisation: this requires
  **dynamic** curve specification later, at price time.
- **With** specifying all pricing ``curves`` at initialisation: this requires curves to
  exist as objects, either ``Curve`` or ``LineCurve`` objects.
- **Indirectly** specifying all pricing ``curves`` by known string ``id``. This does
  not require any curves to pre-exist.

At price time there are then also **three modes of pricing an**
``Instrument``. The signature of the ``rate``, ``npv`` and ``cashflows`` methods
contains the arguments; ``curves`` and ``solver``.

- If ``curves`` are given dynamically these are used regardless of which initialisation
  mode was used. The input here will **overwrite** the curves specified at
  initialisation.
- If ``curves`` are not given dynamically then those ``curves`` provided at
  initialisation will be used.

  - If they were provided as objects these are used directly.
  - If they were provided as string ``id`` form, then a ``solver`` is required
    from which the relevant curves will be extracted.

If ``curves`` are given dynamically in combination with a ``solver``, and those curves
do not form part of the solver's iteration remit then depending upon the options,
errors or warnings might be raised or ignore. See XXXXX TODO


Best Practice
***************

The recommended way of working within ``rateslib``
is to initialise ``Instruments`` with a defined ``curves`` argument
**as string** ``id`` s. This does not
impede dynamic pricing if ``curves`` are constructed and supplied later directly to
pricing methods.
The ``curves`` attribute on the ``Instrument`` is instructive of its pricing intent.

.. ipython:: python

   irs = IRS(
       effective=dt(2022, 1, 1),
       termination="6m",
       frequency="Q",
       currency="usd",
       notional=500e6,
       fixed_rate=2.0,
       curves="sofr",  # or ["sofr", "sofr"] for forecasting and discounting
   )
   irs.curves

At any point a ``Curve`` could be constructed and used for dynamic pricing, even if
its ``id`` does not match the instrument initialisation.

.. ipython:: python

   curve = Curve(
       nodes={dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98},
       id="not_sofr"
   )
   irs.rate(curve)

The above output would have resulted regardless of under which of the three
modes the ``Instrument`` was initialised under; without ``curves``,  with ``curves``, or
using indirect ``id`` s.

Why is this best practice?
---------------------------

The reasons that this is best practice are:

- It provides more flexibility when working with multiple different curve models and
  multiple :class:`~rateslib.solver.Solver` s.
- It provides more flexibility since only ``Instruments`` constructed in this manner
  can be directly added to the :class:`~rateslib.instruments.Portfolio` class. It also
  extends the :class:`~rateslib.instruments.Spread` and
  :class:`~rateslib.instruments.Fly` classes.
- It creates redundancy by avoiding programmatic errors when curves are overwritten and
  object oriented associations are silently broken, which can occur when using the
  other methods.
- It is anticipated that this mechanism is the one most future proofed when ``rateslib``
  is extended for server-client-api transfer via JSON.

Multiple curve model ``Solver`` s
---------------------------------

Consider two different curve models, a **log-linear** one and a **log-cubic spline**,
which we calibrate with the same instruments.

.. ipython:: python

   instruments = [
       IRS(dt(2022, 1, 1), "4m", "Q", curves="sofr"),
       IRS(dt(2022, 1, 1), "8m", "Q", curves="sofr"),
   ]
   s = [1.85, 2.10]
   ll_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 5, 1): 1.0,
           dt(2022, 9, 1): 1.0
       },
       interpolation="log_linear",
       id="sofr"
   )
   lc_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 5, 1): 1.0,
           dt(2022, 9, 1): 1.0
       },
       t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
          dt(2022, 5, 1),
          dt(2022, 9, 1), dt(2022, 9, 1), dt(2022, 9, 1), dt(2022, 9, 1)],
       id="sofr",
   )
   ll_solver = Solver(curves=[ll_curve], instruments=instruments, s=s, instrument_labels=["4m", "8m"], id="sofr")
   lc_solver = Solver(curves=[lc_curve], instruments=instruments, s=s, instrument_labels=["4m", "8m"], id="sofr")
   ll_curve.plot("1D", comparators=[lc_curve], labels=["LL Curve", "LC Curve"])

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import IRS
   from rateslib.solver import Solver
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   instruments = [
       IRS(dt(2022, 1, 1), "4m", "Q", curves="sofr"),
       IRS(dt(2022, 1, 1), "8m", "Q", curves="sofr"),
   ]
   s = [1.85, 2.10]
   ll_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 5, 1): 1.0,
           dt(2022, 9, 1): 1.0
       },
       interpolation="log_linear",
       id="sofr"
   )
   lc_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 5, 1): 1.0,
           dt(2022, 9, 1): 1.0
       },
       t=[dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1), dt(2022, 1, 1),
          dt(2022, 5, 1),
          dt(2022, 9, 1), dt(2022, 9, 1), dt(2022, 9, 1), dt(2022, 9, 1)],
       id="sofr",
   )
   ll_solver = Solver(curves=[ll_curve], instruments=instruments, s=s)
   lc_solver = Solver(curves=[lc_curve], instruments=instruments, s=s)
   fig, ax, line = ll_curve.plot("1D", comparators=[lc_curve], labels=["Log-Linear", "Log_Cubic"])
   plt.show()

Since the ``irs`` instrument was initialised indirectly with string ``id`` s we can
supply the ``Solver`` s as pricing parameters and the curves named *"sofr"* in each
of them will be looked up and used to price the ``irs``.

.. ipython:: python

   irs.rate(solver=ll_solver)
   irs.rate(solver=lc_solver)

The :class:`~rateslib.dual.Dual` datatypes already hint at different risk sensitivities
of the instrument under the different curve model solvers. For good order we can
display the delta risks.

.. ipython:: python

   irs.delta(solver=ll_solver)
   irs.delta(solver=lc_solver)

Using a ``Portfolio``
----------------------

We can consider creating another ``Solver`` for the ESTR curve which extends the SOFR
solver.

.. ipython:: python

   instruments = [
       IRS(dt(2022, 1, 1), "3m", "Q", curves="estr"),
       IRS(dt(2022, 1, 1), "9m", "Q", curves="estr"),
   ]
   s = [0.75, 1.65]
   ll_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2022, 4, 1): 1.0,
           dt(2022, 10, 1): 1.0
       },
       interpolation="log_linear",
       id="estr",
   )
   combined_solver = Solver(
       curves=[ll_curve],
       instruments=instruments,
       s=s,
       instrument_labels=["3m", "9m"],
       pre_solvers=[ll_solver],
       id="estr"
   )

Now we create another :class:`~rateslib.instruments.IRS` and add it to a
:class:`~rateslib.instruments.Portfolio`

.. ipython:: python

   irs2 = IRS(
       effective=dt(2022, 1, 1),
       termination="6m",
       frequency="Q",
       currency="eur",
       notional=-300e6,
       fixed_rate=1.0,
       curves="estr",
   )
   pf = Portfolio([irs, irs2])
   pf.npv(solver=combined_solver)
   pf.delta(solver=combined_solver)
   pf.gamma(solver=combined_solver)


Warnings
*************

Silently breaking object associations
---------------------------------------

.. warning::

   There is no redundancy for breaking object oriented associations when an
   ``Instrument`` is initialised with ``curves`` as objects.

When an ``Instrument`` is created with a **direct object
association** to ``Curves`` which have already been constructed. These will then be
used by default when pricing.

.. ipython:: python

   curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98})
   irs = IRS(dt(2022, 1, 1), "6m", "Q", currency="usd", fixed_rate=2.0, curves=curve)
   irs.rate()
   irs.npv()

If the object is overwritten, or is recreated (say, as a new ``Curve``) the results
will not be as expected.

.. ipython:: python

   curve = "bad_object"  # overwrite the curve variable but the object still exists.
   irs.rate()

It is required to **update** objects instead of recreating them. The documentation
for :meth:`FXForwards.update()<rateslib.fx.FXForwards.update>` also elaborates
on this point.

Disassociated objects
----------------------

.. warning::
   Combining ``curves`` and ``solver`` that are not associated is bad practice. There
   are options for trying to avoid this behaviour.

Consider the below example, which includes two :class:`~rateslib.curves.Curve` s
and a :class:`~rateslib.solver.Solver`.
One :class:`~rateslib.curves.Curve`, labelled "ibor", is **independent**, the other,
labelled "rfr", is associated with the :class:`~rateslib.solver.Solver`, since it has
been iteratively solved.

.. ipython:: python

   rfr_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.98}, id="rfr")
   ibor_curve = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.97}, id="ibor")
   solver = Solver(
       curves=[rfr_curve],
       instruments=[(Value(dt(2023, 1, 1)), ("rfr",), {})],
       s=[0.9825]
   )

When the option ``curve_not_in_solver`` is set to `"ignore"` the independent
:class:`~rateslib.curves.Curve` and a disassociated :class:`~rateslib.solver.Solver`
can be provided to a pricing method and the output returns. It uses the ``curve`` and,
effectively, ignores the disassociated ``solver``.

.. ipython:: python

   irs = IRS(dt(2022, 1, 1), dt(2023, 1, 1), "A")
   defaults.curve_not_in_solver = "ignore"
   irs.rate(ibor_curve, solver)

In the above the ``solver`` is not used for pricing, since it is decoupled from
``ibor_curve``. It is technically an error to list it as an argument.

Setting the option to `"warn"` or `"raise"` enforces a :class:`UserWarning` or a
:class:`ValueError` when this behaviour is detected.

.. .. ipython:: python
      :okwarning:

      defaults.curve_not_in_solver = "warn"
      irs.rate(ibor_curve, solver)

.. ipython:: python
   :okexcept:

   defaults.curve_not_in_solver = "raise"
   try:
       irs.rate(ibor_curve, solver)
   except Exception as e:
       print(e)

When referencing objects by ``id`` s this becomes immediately apparent since, the
below will always fail regardless of the configurable option (the ``solver`` does not
contain the requested curve and therefore cannot fulfill the request).

.. ipython:: python
   :okexcept:

   defaults.curve_not_in_solver = "ignore"
   try:
       irs.rate("ibor", solver)
   except Exception as e:
       print(e)
