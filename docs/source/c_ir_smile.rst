.. _c-ir-smile-doc:

.. ipython:: python
   :suppress:

   from rateslib.volatility import IRSabrSmile, IRSabrCube
   from rateslib.instruments import IRS, IRCall, IRPut, IRStraddle
   from rateslib.curves import Curve
   from rateslib.solver import Solver
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   from pandas import DataFrame

*********************************
IR Vol Smiles & Cubes
*********************************

The ``rateslib.volatility`` module includes classes for *Smiles* and *Cubes*
which can be used to price *IR Options* and *IR Option Strategies*.

.. autosummary::
   rateslib.volatility.IRSabrSmile
   rateslib.volatility.IRSabrCube

Introduction and IR Volatility Smiles
*************************************

A standard for interest rate (IR) options is to utilise a volatility based on SABR parameters.
These are the elements provided in the object's ``nodes``.
An :class:`~rateslib.volatility.IRSabrSmile` is a *Smile* parametrised by the
conventional :math:`\alpha, \beta, \rho, \nu` variables of the SABR model. The parameter
:math:`\beta` is considered a hyper-parameter and will not be varied by a
:class:`~rateslib.solver.Solver` but :math:`\alpha, \rho, \nu` will be varied.

This object must also be initialised with:

- An ``eval_date`` which serves the same purpose as the initial node point on a *Curve*,
  and indicates *'today'* or *'horizon'*. It may be used to determine time to expiry.
- An ``expiry``, for which options priced with this *Smile* must have an equivalent
  expiry or errors will be raised.
- A ``tenor`` indicating the maturity of the underlying :class:`~rateslib.instruments.IRS` that
  this *Smile* will derive.
- An ``irs_series``, which contains the :class:`~rateslib.data.fixings.IRSSeries` conventions for
  defining the mid-market rate on the underlying :class:`~rateslib.instruments.IRS`.

An example of an *IRSabrSmile* is shown below.

.. ipython:: python

   smile = IRSabrSmile(
       eval_date=dt(2000, 1, 1),
       expiry=dt(2000, 7, 1),
       tenor="1y",
       irs_series="usd_irs",
       nodes={
           "alpha": 0.20,
           "beta": 0.5,
           "rho": -0.05,
           "nu": 0.65,
       },
   )
   #  -->  smile.plot(f=2.25,)
   #  -->  smile.plot(x_axis="moneyness", f=2.25)
   #  -->  smile.plot(f=2.25, y_axis="normal_vol")
   #  -->  smile.plot(x_axis="moneyness", f=2.25, y_axis="normal_vol")

Note that *Black Vol* plotted below uses the ``shift`` parameter that is associated with
the *Smile*, but that shift is not plotted. A *Smile* with a larger ``shift`` typically results
in a lower *Black Vol*.

.. container:: twocol

   .. container:: leftside50

      **Strike vs Black Vol Plot**

      .. plot::

         from rateslib.volatility import IRSabrSmile
         from datetime import datetime as dt
         smile = IRSabrSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             tenor="1y",
             irs_series="usd_irs",
             nodes={
                 "alpha": 0.20,
                 "beta": 0.5,
                 "rho": -0.05,
                 "nu": 0.65,
             },
         )
         fig, ax, lines = smile.plot(x_axis="strike", f=2.25)
         plt.show()
         plt.close()

      **Strike vs Normal Vol Plot**

      .. plot::

         from rateslib.volatility import IRSabrSmile
         from datetime import datetime as dt
         smile = IRSabrSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             tenor="1y",
             irs_series="usd_irs",
             nodes={
                 "alpha": 0.20,
                 "beta": 0.5,
                 "rho": -0.05,
                 "nu": 0.65,
             },
         )
         fig, ax, lines = smile.plot(x_axis="strike", f=2.25, y_axis="normal_vol")
         plt.show()
         plt.close()

   .. container:: rightside50

      **Moneyness vs Black Vol Plot**

      .. plot::

         from rateslib.volatility import IRSabrSmile
         from datetime import datetime as dt
         smile = IRSabrSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             tenor="1y",
             irs_series="usd_irs",
             nodes={
                 "alpha": 0.20,
                 "beta": 0.5,
                 "rho": -0.05,
                 "nu": 0.65,
             },
         )
         fig, ax, lines = smile.plot(x_axis="moneyness", f=2.25)
         plt.show()
         plt.close()

      **Moneyness vs Normal Vol Plot**

      .. plot::

         from rateslib.volatility import IRSabrSmile
         from datetime import datetime as dt
         smile = IRSabrSmile(
             eval_date=dt(2000, 1, 1),
             expiry=dt(2000, 7, 1),
             tenor="1y",
             irs_series="usd_irs",
             nodes={
                 "alpha": 0.20,
                 "beta": 0.5,
                 "rho": -0.05,
                 "nu": 0.65,
             },
         )
         fig, ax, lines = smile.plot(x_axis="moneyness", f=2.25, y_axis="normal_vol")
         plt.show()
         plt.close()

.. _c-ir-smile-constructing-doc:

Calibrating a Smile
**********************

It is expected that *Smiles* will typically be calibrated to market prices, similar to
interest rate *Curves*.

As usual with the :class:`~rateslib.solver.Solver` you can use any relevant *Instruments*
and ``metrics`` to
calibrate a *Smile* provided they are sufficiently suitable. Below we use an *ATM-Straddle* and
an *OTM-Put* and *OTM-Call* to define the smile and skew. The normal-bps prices (and note
that the ``metric`` of the *IROption Instruments* are specifically set to *'normal_vol'*) are
calibrated to 50bps, 62bps and 60bps respectively.

.. ipython:: python

   # Define an interest rate curve for pricing the mid-market rate of an IRS
   curve = Curve(
       nodes={dt(2026, 3, 2): 1.0, dt(2029, 3, 2): 0.90},
       calendar="nyc",
       convention="act360",
       id="sofr",
   )
   curve_solver = Solver(
       curves=[curve],
       instruments=[IRS(dt(2026, 3, 4), "2y", spec="usd_irs", curves=["sofr"])],
       s=[3.90],
       instrument_labels=["US_2y"],
   )
   # Define the SABR Smile for pricing a 6m1y Swaption.
   smile = IRSabrSmile(
       eval_date=dt(2026, 3, 2),
       expiry="6m",
       tenor="1y",
       irs_series="usd_irs",
       nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5},
       id="sofr_vol",
   )
   solver = Solver(
       pre_solvers=[curve_solver],  # <- contains the US SOFR Curve
       curves=[smile],              # <- mutates only the smile
       instruments=[
           IRStraddle(dt(2026, 9, 2), "1y", "atm", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
           IRPut(dt(2026, 9, 2), "1y", "-20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
           IRCall(dt(2026, 9, 2), "1y", "+20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
       ],
       s=[50, 62, 60],
       instrument_labels=["ATM", "-20bps", "20bps"],
       id="sofr_sv",
   )

.. plot::
   :caption: Rateslib IR SABR Vol Smile

   from rateslib.curves import Curve
   from rateslib.instruments import *
   from rateslib.volatility import IRSabrSmile
   from rateslib.solver import Solver
   from datetime import datetime as dt
   import matplotlib.pyplot as plt
   # Define an interest rate curve for pricing the mid-market rate of an IRS
   curve = Curve(
       nodes={dt(2026, 3, 2): 1.0, dt(2029, 3, 2): 0.90},
       calendar="nyc",
       convention="act360",
       id="sofr",
   )
   # Define the SABR Smile for pricing a 6m1y Swaption.
   smile = IRSabrSmile(
       eval_date=dt(2026, 3, 2),
       expiry="6m",
       tenor="1y",
       irs_series="usd_irs",
       nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5},
       id="sofr_vol",
   )
   # Solve Curve and Smile simultaneously in this small system
   solver = Solver(
       curves=[curve, smile],
       instruments=[
           IRS(dt(2026, 3, 2), "2y", spec="usd_irs", curves="sofr"),
           IRStraddle(dt(2026, 9, 2), "1y", "atm", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
           IRPut(dt(2026, 9, 2), "1y", "-20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
           IRCall(dt(2026, 9, 2), "1y", "+20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
       ],
       s=[3.90, 50, 62, 60],
       id="sofr_sv",
   )
   fig, ax, line = smile.plot(x_axis="strike", y_axis="normal_vol", labels=["Normal Vol"], curves=curve)
   plt.show()
   plt.close()

IR Volatility Cube
*********************

*IR Swaptions* are typically priced by an object that simultaneously measures its **expiry**,
**tenor** and **strike**. *Rateslib* implements an :class:`~rateslib.volatility.IRSabrCube` which
constructs an individual :class:`~rateslib.volatility.IRSabrSmile` for each **(expiry, tenor)**
pair.

.. container:: twocol

   .. container:: leftside40

      .. image:: _static/ir_sabr_cube_grid.png
         :align: center
         :alt: IR Sabr Cube framework in Rateslib
         :height: 180
         :width: 380

   .. container:: rightside60

      The SABR parameters for each :class:`~rateslib.volatility.IRSabrSmile` are either provided
      directly at each node in the (expiry, tenor) surface grid, or are **bilinearly interpolated**.

      The below example has 4 grid points defined by 2 expiries and 2 tenors. Since each *Smile*
      at each gridpoint has 3 variable SABR parameters this *Cube* has a total of 12 parameters.

.. ipython:: python

   # Define the SABR Cube with 4 gridpoints and consistent SABR parameters.
   cube = IRSabrCube(
       eval_date=dt(2026, 3, 2),
       expiries=["1y", "5y"],
       tenors=["1y", "5y"],
       irs_series="usd_irs",
       beta=0.5,
       alphas=0.25, # <- set every alpha to 0.25 initially
       rhos=-0.05,  # <- set every rho to -0.05 initially
       nus=0.5,     # <- set every nu to 0.50 initially
       id="sofr_cube",
   )
   # Calibrate the Cube accoridng to 12 instruments
   op_args = dict(
       eval_date=dt(2026, 3, 2),
       curves=["sofr"],
       vol="sofr_cube",
       metric="normal_vol",
   )
   instruments = [
       IRStraddle("1y", "1y", "atm", "usd_irs", **op_args),
       IRStraddle("5y", "1y", "atm", "usd_irs", **op_args),
       IRStraddle("1y", "5y", "atm", "usd_irs", **op_args),
       IRStraddle("5y", "5y", "atm", "usd_irs", **op_args),
       IRPut("1y", "1y", "-20bps", "usd_irs", **op_args),
       IRPut("5y", "1y", "-20bps", "usd_irs", **op_args),
       IRPut("1y", "5y", "-20bps", "usd_irs", **op_args),
       IRPut("5y", "5y", "-20bps", "usd_irs", **op_args),
       IRCall("1y", "1y", "+20bps", "usd_irs", **op_args),
       IRCall("5y", "1y", "+20bps", "usd_irs", **op_args),
       IRCall("1y", "5y", "+20bps", "usd_irs", **op_args),
       IRCall("5y", "5y", "+20bps", "usd_irs", **op_args),
   ]
   solver = Solver(
       pre_solvers=[curve_solver],
       surfaces=[cube],
       instruments=instruments,
       s=[
           51.6, 50.8, 55.1, 88.1,  # <- Straddles
           52.1, 51.1, 55.6, 88.8,  # <- OTM Puts
           52.7, 51.5, 55.6, 89.1   # <- OT< Calls
       ],
       instrument_labels=[
           "1y1y ATM", "5y1y ATM", "1y5y ATM", "5y5y ATM",
           "1y1y Put", "5y1y Put", "1y5y Put", "5y5y Put",
           "1y1y Call", "5y1y Call", "1y5y Call", "5y5y Call",
       ]
   )

The solved parameters are visible as attributes on the object.

.. ipython:: python

   cube.alpha_float

.. ipython:: python

   cube.rho_float

.. ipython:: python

   cube.nu_float

Different Volatility Measures
********************************

The SABR model outputs a Black (log-normal) volatility, used in the Black-76 pricing model.
To handle negative interest rates, industry standard is to apply shifts to the strikes and
forwards, e.g. +100bps. This is available as a *meta* property on the pricing object.

Normal basis points volatility, which prices options under the Bachelier pricing model
can be observed;

- in *plots* where the Hagan approximation is used.
- in *Instruments* when the *'normal_vol'* ``metric`` is used.

The :class:`~rateslib.volatility.IRSabrSmile` and :class:`~rateslib.volatility.IRSabrCube` are
simply parameter containers so each, with its own hyper-parameters can be calibrated accoridng to
the same *Instrument* specification, and each resultant object results in very similar option
prices.

.. ipython:: python

   smile_args = dict(
       eval_date=dt(2026, 3, 2),
       expiry="6m",
       tenor="1y",
       irs_series="usd_irs",
       id="sofr_vol",
   )
   def smile_solver_factory(smile):
        solver = Solver(
            pre_solvers=[curve_solver],  # <- contains the US SOFR Curve
            curves=[smile],              # <- mutates only the smile
            instruments=[
                IRStraddle(dt(2026, 9, 2), "1y", "atm", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                IRPut(dt(2026, 9, 2), "1y", "-20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                IRCall(dt(2026, 9, 2), "1y", "+20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
            ],
            s=[50, 62, 60],
            instrument_labels=["ATM", "-20bps", "20bps"],
            id="sofr_sv",
        )

.. ipython:: python

   # Define different SABR Smiles
   smile1 = IRSabrSmile(shift=0, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
   smile2 = IRSabrSmile(shift=0, nodes={"beta": 0.75, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
   smile3 = IRSabrSmile(shift=0, nodes={"beta": 0.25, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
   smile4 = IRSabrSmile(shift=100, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
   smile5 = IRSabrSmile(shift=200, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
   # calibrate each smile similarly
   smile_solver_factory(smile1)
   smile_solver_factory(smile2)
   smile_solver_factory(smile3)
   smile_solver_factory(smile4)
   smile_solver_factory(smile5)


.. container:: twocol

   .. container:: leftside50

      **Strike vs Black Vol Plot**

      .. plot::

         from rateslib import dt, Solver, IRS, Curve, IRCall, IRPut, IRSabrSmile, IRStraddle
         curve = Curve(
             nodes={dt(2026, 3, 2): 1.0, dt(2029, 3, 2): 0.90},
             calendar="nyc",
             convention="act360",
             id="sofr",
         )
         curve_solver = Solver(
             curves=[curve],
             instruments=[IRS(dt(2026, 3, 4), "2y", spec="usd_irs", curves=["sofr"])],
             s=[3.90],
             instrument_labels=["US_2y"],
         )
         smile_args = dict(
             eval_date=dt(2026, 3, 2),
             expiry="6m",
             tenor="1y",
             irs_series="usd_irs",
             id="sofr_vol",
         )
         def smile_solver_factory(smile):
             solver = Solver(
                 pre_solvers=[curve_solver],  # <- contains the US SOFR Curve
                 curves=[smile],              # <- mutates only the smile
                 instruments=[
                     IRStraddle(dt(2026, 9, 2), "1y", "atm", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                     IRPut(dt(2026, 9, 2), "1y", "-20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                     IRCall(dt(2026, 9, 2), "1y", "+20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                 ],
                 s=[50, 62, 60],
                 instrument_labels=["ATM", "-20bps", "20bps"],
                 id="sofr_sv",
             )
         smile1 = IRSabrSmile(shift=0, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile2 = IRSabrSmile(shift=0, nodes={"beta": 0.75, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile3 = IRSabrSmile(shift=0, nodes={"beta": 0.25, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile4 = IRSabrSmile(shift=100, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile5 = IRSabrSmile(shift=200, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         # calibrate each smile similarly
         smile_solver_factory(smile1)
         smile_solver_factory(smile2)
         smile_solver_factory(smile3)
         smile_solver_factory(smile4)
         smile_solver_factory(smile5)

         fig, ax, lines = smile1.plot(x_axis="strike", curves=curve, comparators=[smile2, smile3, smile4, smile5], labels=["b=0.5", "b=0.75", "b=0.25", "shift=100", "shift=200"])
         plt.show()
         plt.close()

   .. container:: rightside50

      **Strike vs Normal Vol Plot**

      .. plot::

         from rateslib import dt, Solver, IRS, Curve, IRCall, IRPut, IRSabrSmile, IRStraddle
         curve = Curve(
             nodes={dt(2026, 3, 2): 1.0, dt(2029, 3, 2): 0.90},
             calendar="nyc",
             convention="act360",
             id="sofr",
         )
         curve_solver = Solver(
             curves=[curve],
             instruments=[IRS(dt(2026, 3, 4), "2y", spec="usd_irs", curves=["sofr"])],
             s=[3.90],
             instrument_labels=["US_2y"],
         )
         smile_args = dict(
             eval_date=dt(2026, 3, 2),
             expiry="6m",
             tenor="1y",
             irs_series="usd_irs",
             id="sofr_vol",
         )
         def smile_solver_factory(smile):
             solver = Solver(
                 pre_solvers=[curve_solver],  # <- contains the US SOFR Curve
                 curves=[smile],              # <- mutates only the smile
                 instruments=[
                     IRStraddle(dt(2026, 9, 2), "1y", "atm", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                     IRPut(dt(2026, 9, 2), "1y", "-20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                     IRCall(dt(2026, 9, 2), "1y", "+20bps", "usd_irs", curves="sofr", vol="sofr_vol", metric="normal_vol"),
                 ],
                 s=[50, 62, 60],
                 instrument_labels=["ATM", "-20bps", "20bps"],
                 id="sofr_sv",
             )
         smile1 = IRSabrSmile(shift=0, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile2 = IRSabrSmile(shift=0, nodes={"beta": 0.75, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile3 = IRSabrSmile(shift=0, nodes={"beta": 0.25, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile4 = IRSabrSmile(shift=100, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         smile5 = IRSabrSmile(shift=200, nodes={"beta": 0.5, "alpha": 0.2, "rho": -0.05, "nu": 0.5}, **smile_args)
         # calibrate each smile similarly
         smile_solver_factory(smile1)
         smile_solver_factory(smile2)
         smile_solver_factory(smile3)
         smile_solver_factory(smile4)
         smile_solver_factory(smile5)

         fig, ax, lines = smile1.plot(x_axis="strike", y_axis="normal_vol", curves=curve, comparators=[smile2, smile3, smile4, smile5], labels=["b=0.5", "b=0.75", "b=0.25", "shift=100", "shift=200"])
         plt.show()
         plt.close()


Delta and Gamma Risks
***********************

Because these objects are integrated into the :class:`~rateslib.solver.Solver` framework
delta and cross-gamma risks are available in the usual way utilising 1st and 2nd order AD,
with risks expressed relative to the rate ``metric`` of each *Instrument*. Consider a
2y2y ATM Payer Swaption:

.. ipython:: python

   irc = IRCall(
       eval_date=dt(2026, 3, 2),
       expiry="2y",
       tenor="2y",
       strike=3.0,
       notional=100e6,
       irs_series="usd_irs",
       curves="sofr",
       vol="sofr_cube",
       premium=1800000.0,
   )
   irc.npv(solver=solver)
   irc.delta(solver=solver)
   irc.gamma(solver=solver)