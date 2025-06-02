.. _cook-exogenous-doc:

.. ipython:: python
   :suppress:

   from rateslib import FXRates, Curve, Solver, IRS, Dual, Variable, defaults
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame, option_context
   defaults.reset_defaults()
   print(defaults.print())

What are Exogenous Variables and Exogenous Sensitivities?
*****************************************************************

Endogenous variables
---------------------

Being a fixed income library, there are some *variables* that are **endogenous** to *rateslib* -
meaning they are created internally and used throughout its internal calculations. These are
often easy to spot. For example when creating an *FXRates* object you will notice the user input
for FX rate information is just expressed with regular *floats*, but *rateslib* internally creates
dual number exposure to these variables.

.. ipython:: python

   fxr = FXRates({"eurusd": 1.10, "gbpusd": 1.25}, settlement=dt(2000, 1, 1))
   fxr.rate(pair="eurgbp")

Similarly, when building *Curves* and calibrating them with a *Solver*, *rateslib* structures
all its parameters internally, so that it can calculate :meth:`~rateslib.solver.Solver.delta` and
:meth:`~rateslib.solver.Solver.gamma` later without any further user input.

.. ipython:: python

   curve = Curve({dt(2000, 1, 1): 1.0, dt(2001, 1, 1): 1.0}, id="curve")
   solver = Solver(
       curves=[curve],
       instruments=[IRS(dt(2000, 1, 1), "6m", "S", curves=curve)],
       s=[2.50],
       id="solver",
   )
   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="6m",
       frequency="S",
       leg2_frequency="M",
       fixed_rate=3.0,
       notional=5e6
   )
   irs.npv(curves=curve)

Exogneous variables
--------------------

**Exogenous** variables are those created dynamically by a user. The only reason one would typically
do this is to create a baseline for measuring some financial sensitivity.

Start with an innocuous example. Suppose we wanted to capture the sensitivity of the *IRS* above
to its notional. The *notional* is just a linear scaling factor for an *IRS* (and many other
instruments too) so the financial exposure for 1 unit of notional is just its *npv* divided by its
5 million notional.

.. ipython:: python

   irs.npv(curves=curve) / 5e6

But this can also be captured using :meth:`~rateslib.instruments.Sensitivities.exo_delta`.

.. ipython:: python

   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="6m",
       frequency="S",
       leg2_frequency="M",
       fixed_rate=3.0,
       notional=Variable(5e6, ["N"]),  # <-- `notional` is assigned as a Variable: 'N'
       curves="curve",
   )
   data = irs.exo_delta(solver=solver, vars=["N"])
   with option_context("display.float_format", lambda x: '%.6f' % x):
       print(data)

What about capturing the exposure to the ``fixed_rate``? This is already provided by the analytical
function :meth:`~rateslib.instruments.IRS.analytic_delta` but it can be shown. Here, we scale
the result from percentage points to basis points.

.. ipython:: python

   irs.analytic_delta(curve)

.. ipython:: python

   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="6m",
       frequency="S",
       leg2_frequency="M",
       fixed_rate=Variable(3.0, ["R"]),  # <-- `fixed_rate` also assigned as: 'R'
       notional=Variable(5e6, ["N"]),
       curves="curve",
   )
   irs.exo_delta(solver=solver, vars=["N", "R"], vars_scalar=[1.0, 1/100])

Exposure to the ``float_spread``? This is also covered by :meth:`~rateslib.instruments.IRS.analytic_delta`, but anyway..

.. ipython:: python

   irs.analytic_delta(curve, leg=2)

.. ipython:: python

   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="6m",
       frequency="S",
       leg2_frequency="M",
       fixed_rate=Variable(3.0, ["R"]),
       notional=Variable(5e6, ["N"]),
       leg2_float_spread=Variable(0.0, ["z"]),   # <-- `float_spread` also assigned as: 'z'
       curves="curve",
   )
   irs.exo_delta(solver=solver, vars=["N", "R", "z"], vars_scalar=[1.0, 1/100, 1.0])

These calculations are completely independent of each other. The *analytic* varieties are just that,
hand coded functions from manually derived equations. The *exo_delta* function organises and
structures the AD *variables* dynamically into the *Solver* and uses the chain rule for
differentiation.

Difference between ``Variable``, and ``Dual`` and ``Dual2``
------------------------------------------------------------

:class:`~rateslib.dual.Dual` and :class:`~rateslib.dual.Dual2` do not permit binary operations
between themselves because it is inconsistent and impossible to correctly define second order
derivatives with such operations. For safety, *TypeErrors* are raised when this is encountered.
Internally, for specific calculations dual numbers are converted to specific types first
before performing calculations in *rateslib*.

But if a user wants to inject dual sensitivity at an arbitrary point in the code it may not be
possible for *rateslib* to know what to convert and this may break downstream calculations.

The below example shows a user injecting a *Dual2* sensitivity directly and the calculations
breaking becuase other variable are only in *Dual* mode.

.. ipython:: python

   irs = IRS(
       effective=dt(2000, 1, 1),
       termination="6m",
       frequency="S",
       leg2_frequency="M",
       fixed_rate=Dual2(3.0, ["R"], [], []),  # <-- `fixed_rate` added as a Dual2
       curves="curve",
   )
   try:
       irs.delta(solver=solver)
   except TypeError as e:
       print(e)

Using a :class:`~rateslib.dual.Variable`, instead, is designed to cover these user cases.
A *Variable* will convert to the necessary type as and when the calculation requires.

The Real Use Case
-------------------

The use case that triggered the development of **exogenous** variables came with
credit default swaps (:class:`~rateslib.instruments.CDS`). If you go through the
:ref:`Replicating a Pfizer Default Curve and CDS <cook-cdsw-doc>` cookbook page, right at the
very bottom in the Bloomberg screenshot is a calculated figure::

  Rec Risk (1%): 78.75

This is the financial exposure of the constructed *CDS* if the recovery rate of Pfizer CDSs
increase by 1%. But, the
nuanced aspect of this value is that it is not what happens if the recovery rate of the
specifically constructed *CDS* changes in recovery rate (that is very easy to measure), but
rather what
happens if Pfizer's overall recovery rate changes for all its CDSs. This impacts all of
the calibrating
instruments used in the construction of the hazard *Curve*, and by implication all of the
gradients attached to the *Solver*.

We will replicate all of the code from that page, some of the variables are directly shown:

.. ipython:: python
   :suppress:

   from rateslib import add_tenor, CDS
   irs_tenor = ["1m", "2m", "3m", "6m", "12m", "2y", "3y", "4y", "5y", "6y", "7y", "8y", "9y", "10y", "12y"]
   irs_rates = [4.8457, 4.7002, 4.5924, 4.3019, 3.8992, 3.5032, 3.3763, 3.3295, 3.3165, 3.3195, 3.3305, 3.3450, 3.3635, 3.3830, 3.4245]
   cds_tenor = ["6m", "12m", "2y", "3y", "4y", "5y", "7y", "10y"]
   cds_rates = [0.11011, 0.14189, 0.20750, 0.26859, 0.32862, 0.37861, 0.51068, 0.66891]
   today = dt(2024, 10, 4)  # Friday 4th October 2024
   spot = dt(2024, 10, 8)  # Tuesday 8th October 2024
   disc_curve = Curve(
       nodes={
           today: 1.0,
           **{add_tenor(spot, _, "mf", "nyc"): 1.0 for _ in irs_tenor}
       },
       calendar="nyc",
       convention="act360",
       interpolation="log_linear",
       id="sofr"
   )
   us_rates_sv = Solver(
       curves=[disc_curve],
       instruments=[
           IRS(spot, _, spec="usd_irs", curves="sofr") for _ in irs_tenor
       ],
       s=irs_rates,
       instrument_labels=irs_tenor,
       id="us_rates"
   )
   cds_eff = dt(2024, 9, 20)
   cds_mats = [add_tenor(dt(2024, 12, 20), _, "mf", "all") for _ in cds_tenor]

   hazard_curve = Curve(
       nodes={
           today: 1.0,
           **{add_tenor(spot, _, "mf", "nyc"): 1.0 for _ in cds_tenor}
       },
       calendar="all",
       convention="act365f",
       interpolation="log_linear",
       credit_recovery_rate=0.4,
       id="pfizer"
   )
   pfizer_sv = Solver(
       curves=[hazard_curve],
       pre_solvers=[us_rates_sv],
       instruments=[
           CDS(
               effective=cds_eff,
               termination=_,
               spec="us_ig_cds",
               curves=["pfizer", "sofr"]
           ) for _ in cds_mats
       ],
       s=cds_rates,
       instrument_labels=cds_tenor,
       id="pfizer_cds"
   )

.. ipython:: python

   disc_curve  # the US SOFR discount curve created
   us_rates_sv  # the Solver calibrating the SOFR curve
   hazard_curve  # the Pfizer hazard curve
   pfizer_sv  # the Solver calibrating the hazard curve

First we can demonstrate what happens when we inject sensitivity directly to a single
*Instrument* calculation. The :meth:`~rateslib.instruments.CDS.analytic_rec_risk` is an
analytic calculation that determines the change in value for a 1% change in recovery rate
just for a single *CDS Instrument*

.. ipython:: python

   cds = CDS(
       effective=dt(2024, 9, 20),
       termination=dt(2029, 12, 20),
       spec="us_ig_cds",
       curves=["pfizer", "sofr"],
       notional=10e6,
   )
   cds.analytic_rec_risk(hazard_curve, disc_curve)

We can also obtain this value by copying a curve, injecting sensitivity, as an
**exogenous variable** into it and evaluating with
:meth:`~rateslib.instruments.Sensitivities.exo_delta`. This copied curve is independent from,
and not mapped by, the *Solver* so none of the *Solver's* parameters are made sensitive to the
change in ``recovery_rate`` here.

.. ipython:: python

   hazard_curve_copy = hazard_curve.copy()
   # Set a new id to avoid Solver curve mapping errors
   hazard_curve_copy._id = "something_else"
   # Inject sensitivity to the recovery rate
   hazard_curve_copy.update_meta("credit_recovery_rate", Variable(0.4, ["RR"]))
   cds.exo_delta(curves=[hazard_curve_copy, disc_curve], solver=pfizer_sv, vars=["RR"], vars_scalar=[1/100.0])

But this isn't really the value we want to capture. In fact we want to capture the change in NPV
when the recovery rate of **all** Pfizer CDSs (including those that are calibrating the curves)
have a different recovery rate, i.e. when the ``recovery_rate`` on the original
``hazard_curve`` is updated. This is the same process, except this time we inject the
sensitivity to the *Solver's* mapped curve directly and re-iterate.

.. ipython:: python

   # Update the Pfizer hazard curve to have exogenous exposure to "RR" variable
   hazard_curve.update_meta("credit_recovery_rate", Variable(0.40, ["RR"]))
   pfizer_sv.iterate()
   cds.exo_delta(solver=pfizer_sv, vars=["RR"], vars_scalar=[1/100.0])

This value is close to BBG's estimate of 78.75. But let's validate it by
resorting (just this once!) to numerical differentiation and see what happens there:

1) Record the initial values:

.. ipython:: python

   base_npv = cds.npv(solver=pfizer_sv)
   base_npv

2) Update the ``recovery_rate`` parameter and re-iterate the solver:

.. ipython:: python

   hazard_curve.update_meta("credit_recovery_rate", 0.41)
   pfizer_sv.iterate()

3) Revalue the NPV and compare it with the previous base value, scaling for 1% RR.

.. ipython:: python

   fwd_diff = cds.npv(solver=pfizer_sv)
   float(fwd_diff - base_npv)

Personally, I am inclined to trust *rateslib's* own figures here, rather than BBG, since these
are calculated using AD and analytical maths and supported by a comparison to a forward
difference method.
