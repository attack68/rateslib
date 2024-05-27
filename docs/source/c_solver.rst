.. _c-solver-doc:

.. ipython:: python
   :suppress:

   import warnings
   warnings.filterwarnings('always')
   from rateslib.solver import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np

***********
Solver
***********

The ``rateslib.solver`` module includes a :class:`~rateslib.solver.Solver` class
which iteratively solves for the parameters of :ref:`Curve <curves-doc>` objects, to
fit the given market data of calibrating :ref:`Instruments <instruments-toc-doc>`.

This module relies on the utility module :ref:`dual<dual-doc>` for gradient based
optimization.

.. autosummary::
   rateslib.solver.Solver

Parameters
***********

The :class:`~rateslib.solver.Solver` solves the following least squares
objective function:

.. math::

   \min_\mathbf{v} f(\mathbf{v}, \mathbf{S}) = (\mathbf{r(v)-S})\mathbf{W}(\mathbf{r(v)-S})^\mathbf{T}

where :math:`\mathbf{S}` are the known calibrating instrument rates,
:math:`\mathbf{r}` are the determined instrument rates based on the solved parameters,
:math:`\mathbf{v}`, and :math:`\mathbf{W}` is a diagonal matrix of weights.

Each curve type has the following parameters:

.. list-table:: Parameters and hyper parameters of Curves and Solver interaction.
   :widths: 15 15 35 35
   :header-rows: 1

   * - Parameter
     - Type
     - Summary
     - Affected by ``Solver``
   * - ``interpolation``
     - Hyper parameter
     - Equation or mechanism to determine intermediate values not defined explicitly
       by ``nodes``.
     - No
   * - ``nodes`` keys
     - Hyper parameters
     - Fixed points which implicitly impact the interpolated values across the curve.
     - No
   * - ``t``
     - Hyper parameters
     - Framework for defining the (log) cubic spline structure which implicitly impacts
       the interpolated values across the curve.
     - No
   * - ``endpoints``
     - Hyper parameters
     - Method used to control spline curves on the left and right boundaries.
     - No
   * - ``nodes`` values
     - **Parameters**
     - The explicit values associated with node dates.
     - | **Yes.**
       | For :class:`~rateslib.curves.Curve` all parameters except the initial node value of 1.0 is varied.
       | For :class:`~rateslib.curves.LineCurve` all parameters including the initial node value is varied.


Calibrating Curves
******************

Thus, in order to calibrate or solve curves the hyper parameters must already
be defined, so that ``nodes``, ``interpolation``, ``t`` and ``endpoints`` must all
be configured. These will not be changed by the :class:`~rateslib.solver.Solver`.
The ``nodes`` values (the parameters) should be initialised with sensible values
from which the
optimizer will start. However, it is usually quite robust and should be able to solve
from a variety of initialised node values.

We define a simple :class:`~rateslib.curves.Curve` using default hyper parameters
and only a few ``nodes``.

.. ipython:: python

   ll_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.979,
           dt(2025,1,3): 0.967
       },
       id="curve",
   )

Next, we must define the ``instruments`` which will instruct the solution.

.. ipython:: python

   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
   ]

There are a number of different mechanisms for the way in which this can be done,
but the example here reflects **best practice** as demonstrated in
:ref:`pricing mechanisms<mechanisms-doc>`.

Once a suitable, and valid, set of instruments has been configured we can supply it,
and the curves, to the solver. We must also supply some target rates, ``s``, and
the optimizer will update the curves.

.. ipython:: python

   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = [1.0, 1.6, 2.0],
   )
   ll_curve.plot("1D")

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   ll_curve = Curve(
       nodes={dt(2022,1,1): 1.0, dt(2023,1,1): 0.99, dt(2024,1,1): 0.965, dt(2025,1,1): 0.93},
       interpolation="log_linear",
       id="curve"
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
   ]
   s = np.array([1.0, 1.6, 2.0])
   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = s,
   )
   fig, ax, line = ll_curve.plot("1D")
   plt.show()

The values of the ``solver.s`` can be updated and the curves can be redetermined

.. ipython:: python

   print(instruments[1].rate(ll_curve).real)
   solver.s[1] = 1.5
   solver.iterate()
   print(instruments[1].rate(ll_curve).real)

Changing the hyper parameters of a curve does not require any fundamental
change to the input arguments to the :class:`~rateslib.solver.Solver`.
Here a mixed interpolation scheme is used and the :class:`~rateslib.curves.Curve`
calibrated.

.. ipython:: python

   mixed_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,3): 0.93,
       },
       interpolation="log_linear",
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,3), dt(2025,1,3), dt(2025,1,3), dt(2025,1,3)],
       id="curve",
   )
   solver = Solver(
       curves = [mixed_curve],
       instruments = instruments,
       s = [1.0, 1.5, 2.0],
   )
   ll_curve.plot("1D", comparators=[mixed_curve], labels=["log-linear", "mixed"])

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   ll_curve = Curve(
       nodes={dt(2022,1,1): 1.0, dt(2023,1,1): 0.99, dt(2024,1,1): 0.965, dt(2025,1,3): 0.93},
       interpolation="log_linear",
       id="curve",
   )
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
   ]
   s = np.array([1.0, 1.5, 2.0])
   solver = Solver(
       curves = [ll_curve],
       instruments = instruments,
       s = s,
   )
   spline_curve = Curve(
       nodes={
           dt(2022,1,1): 1.0,
           dt(2023,1,1): 0.99,
           dt(2024,1,1): 0.965,
           dt(2025,1,3): 0.93,
       },
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,3), dt(2025,1,3), dt(2025,1,3), dt(2025,1,3)],
       id="curve",
   )
   solver = Solver(
       curves = [spline_curve],
       instruments = instruments,
       s = s,
   )
   fig, ax, lines = ll_curve.plot("1D", comparators=[spline_curve], labels=["log-linear", "mixed"])
   plt.show()

Algorithms
***********

In the ``defaults`` settings of ``rateslib``, :class:`~rateslib.solver.Solver` uses
a *"levenberg_marquardt"* algorithm.

There is an option to use a *"gauss_newton*" algorithm which is faster if the
initial guess is reasonable. This should be used where possible, but this is a more
unstable algorithm so is not set as the default.

For other debugging procedures the *"gradient_descent"* method is available although
this is not recommended due to computational inefficiency.

Details on these algorithms are provided in the ``rateslib``
:ref:`supplementary materials<about-doc>`.

Weights
********

The argument ``weights`` allows certain instrument rates to be targeted with
greater priority than others. In the above examples this was of no relevance since
in all previous cases the minimum solution of zero was fully attainable.

The following pathological example, where the same instruments are
provided multiple times with different rates, shows the effect.

.. ipython:: python

   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "1Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "2Y", "A", curves="curve"),
       IRS(dt(2022, 1, 1), "3Y", "A", curves="curve"),
   ]
   solver = Solver(
       curves = [mixed_curve],
       instruments = instruments,
       s = [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
       weights = [1, 1, 1, 1e-4, 1e-4, 1e-4],
   )
   for instrument in instruments:
       print(float(instrument.rate(solver=solver)))

   solver = Solver(
       curves = [mixed_curve],
       instruments = instruments,
       s = [1.0, 1.1, 1.2, 5.0, 5.1, 5.2],
       weights = [1e-4, 1e-4, 1e-4, 1, 1, 1],
   )
   for instrument in instruments:
       print(float(instrument.rate(solver=solver)))

Dependency Chains
******************

In real fixed income trading environments every curve should be synchronous and
dependencies should use the same construction method in one division as in another.
The ``pre_solvers`` argument allows a chain of :class:`~rateslib.solver.Solver` s.
Here a SOFR curve is constructed via a solver and is then added to another solver
which solves an ESTR curve. There is no technical dependence here of one on the
other so these solvers could be arranged in either order.

.. ipython:: python

   sofr_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="sofr",
   )
   sofr_instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="sofr"),
       IRS(dt(2022, 1, 1), "2Y", "A", currency="usd", curves="sofr"),
       IRS(dt(2022, 1, 1), "3Y", "A", currency="usd", curves="sofr"),
   ]
   sofr_solver = Solver(
       curves = [sofr_curve],
       instruments = sofr_instruments,
       s = [2.5, 3.0, 3.5],
   )
   estr_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="estr",
   )
   estr_instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves="estr"),
       IRS(dt(2022, 1, 1), "2Y", "A", currency="eur", curves="estr"),
       IRS(dt(2022, 1, 1), "3Y", "A", currency="eur", curves="estr"),
   ]
   estr_solver = Solver(
       curves = [estr_curve],
       instruments = estr_instruments,
       s = [1.25, 1.5, 1.75],
       pre_solvers=[sofr_solver]
   )

It is possible to create only a single solver using the two curves and six instruments
above. However, in practice it is less efficient to solve independent solvers
within the same framework. And practically, this is not usually how trading teams are
configured, all as one big group. Normally siloed teams are responsible for their
own subsections, be it one currency or another, or different product types.

Multi-Currency Instruments
***************************

Multi-currency derivatives rely on :class:`~rateslib.fx.FXForwards`. In this
example we establish a new cash-collateral discount curve and use
:class:`~rateslib.instruments.XCS` within a :class:`~rateslib.solver.Solver`.

.. ipython:: python

   eurusd = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="eurusd",
   )
   fxr = FXRates({"eurusd": 1.10}, settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "eureur": estr_curve,
           "eurusd": eurusd,
           "usdusd": sofr_curve,
       }
   )
   kwargs={
       "currency": "eur",
       "leg2_currency": "usd",
       "curves": ["estr", "eurusd", "sofr", "sofr"],
   }
   xcs_instruments = [
       XCS(dt(2022, 1, 1), "1Y", "A", **kwargs),
       XCS(dt(2022, 1, 1), "2Y", "A", **kwargs),
       XCS(dt(2022, 1, 1), "3Y", "A", **kwargs),
   ]
   xcs_solver = Solver(
       curves = [eurusd],
       instruments = xcs_instruments,
       s = [-10, -15, -20],
       fx=fxf,
       pre_solvers=[estr_solver],
   )
   estr_curve.plot("1d", comparators=[eurusd], labels=["Eur:eur", "Eur:usd"])

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   sofr_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="sofr",
   )
   sofr_instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="sofr"),
       IRS(dt(2022, 1, 1), "2Y", "A", currency="usd", curves="sofr"),
       IRS(dt(2022, 1, 1), "3Y", "A", currency="usd", curves="sofr"),
   ]
   sofr_solver = Solver(
       curves = [sofr_curve],
       instruments = sofr_instruments,
       s = [2.5, 3.0, 3.5],
   )
   estr_curve = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="estr",
   )
   estr_instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves="estr"),
       IRS(dt(2022, 1, 1), "2Y", "A", currency="eur", curves="estr"),
       IRS(dt(2022, 1, 1), "3Y", "A", currency="eur", curves="estr"),
   ]
   estr_solver = Solver(
       curves = [estr_curve],
       instruments = estr_instruments,
       s = [1.25, 1.5, 1.75],
       pre_solvers=[sofr_solver]
   )
   eurusd = Curve(
       nodes={
           dt(2022, 1, 1): 1.0,
           dt(2023, 1, 1): 1.0,
           dt(2024, 1, 1): 1.0,
           dt(2025, 1, 1): 1.0,
       },
       id="eurusd",
   )
   fxr = FXRates({"eurusd": 1.10}, settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "eureur": estr_curve,
           "eurusd": eurusd,
           "usdusd": sofr_curve,
       }
   )
   kwargs={
       "currency": "eur",
       "leg2_currency": "usd",
       "curves": ["estr", "eurusd", "sofr", "sofr"],
   }
   xcs_instruments = [
       XCS(dt(2022, 1, 1), "1Y", "A", **kwargs),
       XCS(dt(2022, 1, 1), "2Y", "A", **kwargs),
       XCS(dt(2022, 1, 1), "3Y", "A", **kwargs),
   ]
   xcs_solver = Solver(
       curves = [eurusd],
       instruments = xcs_instruments,
       s = [-10, -15, -20],
       fx=fxf,
       pre_solvers=[estr_solver],
   )
   fig, ax, lines = estr_curve.plot("1D", comparators=[eurusd], labels=["Eur:eur", "Eur:usd"])
   plt.show()
   plt.close()


Calibration Instrument Error
*****************************

Depending upon the hyper parameters, parameters and calibrating instrument choices,
the optimized solution may well lead to curves that do not completely reprice the
calibrating instruments. Sometimes this is representative of errors in the construction
process, and at other times this is completely desirable.

When the :class:`~rateslib.solver.Solver` is initialised and iterates it will print
an output to console indicating a success or failure and the value of the
objective function. If this value is very small, that already indicates that there is
no error in any instruments. However for cases where the curve is over-specified, error
is to be expected.

.. ipython:: python

   solver_with_error = Solver(
       curves=[
           Curve(
               nodes={dt(2022, 1, 1): 1.0, dt(2022, 7, 1): 1.0, dt(2023, 1, 1): 1.0},
               id="curve1"
           )
       ],
       instruments=[
           IRS(dt(2022, 1, 1), "1M", "A", curves="curve1"),
           IRS(dt(2022, 1, 1), "2M", "A", curves="curve1"),
           IRS(dt(2022, 1, 1), "3M", "A", curves="curve1"),
           IRS(dt(2022, 1, 1), "4M", "A", curves="curve1"),
           IRS(dt(2022, 1, 1), "8M", "A", curves="curve1"),
           IRS(dt(2022, 1, 1), "12M", "A", curves="curve1"),
       ],
       s=[2.0, 2.2, 2.3, 2.4, 2.45, 2.55],
       instrument_labels=["1m", "2m", "3m", "4m", "8m", "12m"],
   )
   solver_with_error.error


Composite, Proxy and Multi-CSA Curves
****************************************

:class:`~rateslib.curves.CompositeCurve`, :class:`~rateslib.curves.ProxyCurve` and
:class:`~rateslib.curves.MultiCsaCurve` do not
have their own parameters. These rely on the parameters from other fundamental curves.
It is possible to create a *Solver* defined with *Instruments* that reference these
complex curves as pricing curves with the *Solver* updating the underlying
parameters of the fundamental curves.

This does not require much additional configuration, it simply requires ensuring
all necessary curves are documented.

Below we will calculate a EUR IRS defined by a *CompositeCurve* and a *Curve*,
a USD IRS defined just by a *Curve*, and then create an :class:`~rateslib.fx.FXForwards`
defined with USD collateral, but calibrate a solver by
:class:`~rateslib.instruments.XCS` instruments priced with EUR collateral.

.. ipython:: python

   eureur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eureur")
   eurspd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.999}, id="eurspd")
   eur3m = CompositeCurve([eureur, eurspd], id="eur3m")
   usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="usdusd")
   eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eurusd")
   fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "eureur": eureur,
           "usdusd": usdusd,
           "eurusd": eurusd,
       }
   )
   usdeur = fxf.curve("usd", "eur", id="usdeur")
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves=["eur3m", "eureur"]),
       IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="usdusd"),
       XCS(dt(2022, 1, 1), "1Y", "A", currency="eur", leg2_currency="usd", curves=["eureur", "eureur", "usdusd", "usdeur"]),
   ]
   solver = Solver(curves=[eureur, eur3m, usdusd, eurusd, usdeur], instruments=instruments, s=[2.0, 2.7, -15], fx=fxf)

We can plot all five curves defined above by the 3 fundamental curves,
*'eureur', 'usdusd', 'eurusd'*.

.. ipython:: python

   eureur.plot("1d", comparators=[eur3m, eurusd], labels=["eureur", "eur3m", "eurusd"])
   usdusd.plot("1d", comparators=[usdeur], labels=["usdusd", "usdeur"])

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   eureur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eureur")
   eurspd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.999}, id="eurspd")
   eur3m = CompositeCurve([eureur, eurspd], id="eur3m")
   usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="usdusd")
   eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eurusd")
   fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "eureur": eureur,
           "usdusd": usdusd,
           "eurusd": eurusd,
       }
   )
   usdeur = fxf.curve("usd", "eur", id="usdeur")
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves=["eur3m", "eureur"]),
       IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="usdusd"),
       XCS(dt(2022, 1, 1), "1Y", "A", currency="eur", leg2_currency="usd", curves=["eureur", "eureur", "usdusd", "usdeur"]),
   ]
   solver = Solver(curves=[eureur, eur3m, usdusd, eurusd, usdeur], instruments=instruments, s=[2.0, 2.7, -15], fx=fxf)
   fig, ax, lines = eureur.plot("1d", comparators=[eur3m, eurusd], labels=["eureur", "eur3m", "eurusd"])
   plt.show()
   plt.close()

.. plot::

   from rateslib.curves import *
   from rateslib.instruments import *
   import matplotlib.pyplot as plt
   from datetime import datetime as dt
   import numpy as np
   eureur = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eureur")
   eurspd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 0.999}, id="eurspd")
   eur3m = CompositeCurve([eureur, eurspd], id="eur3m")
   usdusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="usdusd")
   eurusd = Curve({dt(2022, 1, 1): 1.0, dt(2023, 1, 1): 1.0}, id="eurusd")
   fxr = FXRates({"eurusd": 1.1}, settlement=dt(2022, 1, 3))
   fxf = FXForwards(
       fx_rates=fxr,
       fx_curves={
           "eureur": eureur,
           "usdusd": usdusd,
           "eurusd": eurusd,
       }
   )
   usdeur = fxf.curve("usd", "eur", id="usdeur")
   instruments = [
       IRS(dt(2022, 1, 1), "1Y", "A", currency="eur", curves=["eur3m", "eureur"]),
       IRS(dt(2022, 1, 1), "1Y", "A", currency="usd", curves="usdusd"),
       XCS(dt(2022, 1, 1), "1Y", "A", currency="eur", leg2_currency="usd", curves=["eureur", "eureur", "usdusd", "usdeur"]),
   ]
   solver = Solver(curves=[eureur, eur3m, usdusd, eurusd, usdeur], instruments=instruments, s=[2.0, 2.7, -15], fx=fxf)
   fig, ax, lines = usdusd.plot("1d", comparators=[usdeur], labels=["usdusd", "usdeur"])
   plt.show()
   plt.close()