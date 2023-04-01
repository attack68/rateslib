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
which iteratively solves the DFs of :class:`~rateslib.curves.Curve` objects, and
values of :class:`~rateslib.curves.LineCurve` to fit given
calibrating instruments.

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
           dt(2025,1,1): 0.967
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

   s = np.array([1.0, 1.6, 2.0])
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
           dt(2025,1,1): 0.93,
       },
       interpolation="log_linear",
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1)],
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
       nodes={dt(2022,1,1): 1.0, dt(2023,1,1): 0.99, dt(2024,1,1): 0.965, dt(2025,1,1): 0.93},
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
           dt(2025,1,1): 0.93,
       },
       t = [dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2023,1,1), dt(2024,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1), dt(2025,1,1)],
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
a *"gauss_newton"* algorithm.

In the rare cases that this fails to solve for sensible starting values, or if sensible
starting values are not necessarily known, try using the *"levenberg_marquardt"*
alternative.

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
       calendar="nyc",
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
       calendar="tgt",
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